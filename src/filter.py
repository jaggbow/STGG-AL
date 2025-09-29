import pickle
import subprocess
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import ray
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

ray.init()

FRAGMENTS_DIR = Path("../resource/fragments.pkl")
MAX_RING_SIZE = 6


def get_largest_fused_ring_system(smiles):
    # Extract the largest fused ring system from a given molecule.
    mol = Chem.MolFromSmiles(smiles)
    ri = mol.GetRingInfo()
    if not ri.NumRings():
        return None  # No rings in the molecule

    # Get ring bonds and atoms
    ring_bonds = ri.BondRings()
    ring_atoms = ri.AtomRings()

    # Create a graph where nodes are rings and edges represent shared atoms or bonds
    ring_graph = {}
    for i in range(len(ring_atoms)):
        ring_graph[i] = set()
        for j in range(len(ring_atoms)):
            if i != j:
                # Check if rings share bonds or atoms
                if set(ring_bonds[i]).intersection(ring_bonds[j]) or set(
                    ring_atoms[i]
                ).intersection(ring_atoms[j]):
                    ring_graph[i].add(j)
    # Find the largest connected component of the ring graph

    visited = set()
    largest_component = set()

    def dfs(node, component):
        visited.add(node)
        component.add(node)
        for neighbor in ring_graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in ring_graph:
        if node not in visited:
            current_component = set()
            dfs(node, current_component)
            if len(current_component) > len(largest_component):
                largest_component = current_component

    # Collect atoms and bonds from the largest fused ring component

    largest_ring_atoms = set()
    largest_ring_bonds = set()
    for ring_idx in largest_component:
        largest_ring_atoms.update(ring_atoms[ring_idx])
        largest_ring_bonds.update(ring_bonds[ring_idx])

    # Create a new RWMol containing only the largest fused ring system
    rw_mol = Chem.RWMol()
    atom_mapping = {}  # Map old atom indices to new ones
    for atom_idx in largest_ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        new_idx = rw_mol.AddAtom(atom)
        atom_mapping[atom_idx] = new_idx

    for bond_idx in largest_ring_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if begin_idx in atom_mapping and end_idx in atom_mapping:
            rw_mol.AddBond(
                atom_mapping[begin_idx], atom_mapping[end_idx], bond.GetBondType()
            )
    return Chem.MolToSmiles(rw_mol.GetMol())


def _embed_ff_optimize(mol, workdir, n_confs: int = 50):
    """
    Return path to the lowest-energy MMFF minimized XYZ for *smiles*.
    The conformer search is embarrassingly parallel – RDKit will multithread.
    """
    charge = Chem.GetFormalCharge(mol)
    xyz = workdir / "mmff.xyz"
    if xyz.exists():
        return xyz, charge

    params = AllChem.ETKDGv3()
    params.numThreads = 1
    params.pruneRmsThresh = 0.5
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(ids) == 0:
        return None, None

    energies = []
    for cid in ids:
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid
        )
        if ff is not None:
            ff.Minimize(maxIts=200)
            energies.append((cid, ff.CalcEnergy()))
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            ff.Minimize(maxIts=200)
            energies.append((cid, ff.CalcEnergy()))

    best_cid, best_e = min(energies, key=lambda x: x[1])
    with xyz.open("w") as fh:
        fh.write(f"{mol.GetNumAtoms()}\nMMFF (kcal mol-1): {best_e:.3f}\n")
        conf = mol.GetConformer(best_cid)
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            fh.write(
                f"{atom.GetSymbol():<3} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n"
            )
    return xyz, charge


@ray.remote(num_cpus=1)
def process_smiles(idx, smi, workdir):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol, addCoords=True)
    finaldir = workdir / str(idx)
    finaldir.mkdir(exist_ok=True)

    xyz, charge = _embed_ff_optimize(mol, finaldir)
    if xyz is None:
        return None
    return finaldir, xyz, charge, idx, smi


def _xtb_optimize(
    xyz_in: Path,
    workdir: Path,
    threads: int = 1,
    charge: int = 0,
    xtb_version: str = "2",
    unpaired_e: int = 0,
    xtb_path: str = "xtb",
) -> Path:
    """
    Geometry refine with GFN2-xTB. Returns path to *.xtbopt.xyz*.
    """
    xyz_out = workdir / "geom.xtbopt.xyz"
    if xyz_out.exists():
        return xyz_out
    gfn_version = (
        ["--gfnff"] if str(xtb_version) == "gfnff" else ["--gfn", str(xtb_version)]
    )
    unrestricted = (
        ["--spinpol", "--tblite", f"--uhf {str(unpaired_e)}"] if unpaired_e > 0 else []
    )
    cmd = (
        [
            xtb_path,
            str(xyz_in.name),
            "--opt",
            "--parallel",
            str(threads),
            "--namespace",
            "geom",
            "--charge",
            str(charge),
        ]
        + gfn_version
        + unrestricted
    )
    try:
        with (workdir / "xtb.log").open("w") as log:
            subprocess.run(
                cmd, cwd=workdir, check=True, stdout=log, stderr=subprocess.STDOUT
            )
        if not xyz_out.exists():
            raise RuntimeError("xTB did not produce geom.xtbopt.xyz")
        return xyz_out
    except:
        return None


FRAGMENTS = pickle.load(open(FRAGMENTS_DIR, "rb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type=str)
    parser.add_argument("--property_predictor_ckpt_path", type=str)
    parser.add_argument("--property_predictor_cwd", type=str)
    parser.add_argument("--timeout", type=int, default=75)
    parser.add_argument("--num_threads", type=int, default=2)
    args = parser.parse_args()

    smiles_path = Path(args.smiles_path)
    property_predictor_ckpt_path = args.property_predictor_ckpt_path
    property_predictor_cwd = args.property_predictor_cwd


    existing_smiles = []
    for file in smiles_path.parent.glob("*.pkl"):
        if file != smiles_path:
            old_data = pickle.load(open(file, "rb"))
            if isinstance(old_data, dict) and "smiles" in old_data:
                existing_smiles += old_data["smiles"]
    data = pickle.load(open(smiles_path, "rb"))
    smiles_list = data["smiles"]

    novel_smiles = [smi for smi in smiles_list if smi not in existing_smiles]
    fragment_presence = []
    boron_presence = []
    max_ring_size = []
    nice_smiles = []

    for smiles in tqdm(novel_smiles):
        mol = Chem.MolFromSmiles(smiles)
        core_structure = get_largest_fused_ring_system(smiles)
        fragments_in_mol = core_structure in FRAGMENTS
        fragment_presence.append(int(fragments_in_mol))
        boron_presence.append(
            int(any(atom.GetSymbol() == "B" for atom in mol.GetAtoms()))
        )

        ri = mol.GetRingInfo()
        max_ring_size_ = 0
        if ri.NumRings():
            for ring in ri.AtomRings():
                ring_size = len(ring)
                if ring_size > max_ring_size_:
                    max_ring_size_ = ring_size

        max_ring_size.append(max_ring_size_)

        if (
            fragments_in_mol
            and any(atom.GetSymbol() == "B" for atom in mol.GetAtoms())
            and max_ring_size_ <= MAX_RING_SIZE
        ):
            nice_smiles.append(smiles)

    if len(novel_smiles) > 0:
        print(
            f"There are {len(nice_smiles)} molecules that passed the size and fragment presence check and they represent {(100 * len(nice_smiles) / len(novel_smiles)):.2f} % of the dataset."
        )
    else:
        print("There are no novel smiles.")
        sys.exit()
    data["smiles"] = nice_smiles
    data["statistics"]["num_efficient"] = len(novel_smiles)
    data["statistics"]["num_pass_fragments"] = len(nice_smiles)

    workdir = smiles_path.parent / smiles_path.stem
    workdir.mkdir(exist_ok=True, parents=True)

    futures = [
        process_smiles.remote(idx, smi, workdir) for idx, smi in enumerate(nice_smiles)
    ]
    payload = {}
    with tqdm(total=len(futures)) as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1, timeout=args.timeout)
            if not done:
                continue
            try:
                result = ray.get(done[0], timeout=args.timeout)
            except ray.exceptions.GetTimeoutError:
                ray.cancel(done[0], force=True)
                result = None
            if result is not None:
                finaldir, xyz, charge, idx, smi = result
                payload[f"stgg{idx}"] = {"SMILES": smi}
                xyz_out = _xtb_optimize(xyz, finaldir, threads=args.num_threads, charge=charge)
                if xyz_out is not None:
                    payload[f"stgg{idx}"]["xtb_coordinates_path"] = (
                        xyz_out.absolute().as_posix()
                    )
            pbar.update(1)

    print(
        f"Finished generating molecule coordinates using XTB. You'll find them in {workdir} !"
    )

    # Run property prediction
    csv_fname = smiles_path.parent / f"{smiles_path.stem}.csv"
    if not csv_fname.exists():
        print(f"{csv_fname} doesn't exist, so I will be running the inference.")
        command = [
            "uv",
            "run",
            "--active",
            "predict.py",
            property_predictor_ckpt_path,
            workdir.absolute().as_posix(),
            csv_fname.absolute().as_posix(),
        ]
        with (smiles_path.parent / f"{smiles_path.stem}.log").open("w") as log:
            subprocess.run(
                command,
                cwd=property_predictor_cwd,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        df = pd.read_csv(csv_fname)
        df["SMILES"] = df["id"].apply(lambda x: payload[x]["SMILES"])
        df["xtb_coordinates_path"] = df["id"].apply(
            lambda x: payload[x]["xtb_coordinates_path"]
        )
        df.to_csv(csv_fname)
    else:
        print(
            f"{csv_fname} exists, so I will skip inference and go straight to filtering."
        )

    df = pd.read_csv(csv_fname)
    good_props = df[(df["s1"] < 2.8) & (df["s1"] > 2.6) & (df["delta"] < 0.2)]
    good_idx = [int(item[4:]) for item in good_props["id"]]
    print(
        f"There are {len(good_idx)} molecules that passed the property check and they represent {(100 * len(good_idx) / len(smiles_list)):.2f} % of the dataset."
    )
    data["statistics"]["num_pass_property"] = len(good_idx)
    print(data["statistics"])
    df = df[(df["s1"] < 2.8) & (df["s1"] > 2.6) & (df["delta"] < 0.2)]
    df.to_csv(csv_fname)
    with open(smiles_path, "wb") as file:
        pickle.dump(data, file)
