import argparse
import pickle
import subprocess
from pathlib import Path

import ray
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

ray.init()


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
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                ff.Minimize(maxIts=200)
                energies.append((cid, ff.CalcEnergy()))
            except:
                pass
    if len(energies) == 0:
        return None, None
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
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type=str)
    parser.add_argument("--timeout", type=int, default=75)
    parser.add_argument("--num_threads", type=int, default=2)
    args = parser.parse_args()

    smiles_path = Path(args.smiles_path)
    data = pickle.load(open(smiles_path, "rb"))
    smiles_list = data["smiles"]

    workdir = smiles_path.parent / smiles_path.stem
    workdir.mkdir(exist_ok=True, parents=True)

    futures = [
        process_smiles.remote(idx, smi, workdir) for idx, smi in enumerate(smiles_list)
    ]
    if (smiles_path.parent / f"matcher_{smiles_path.stem}.pkl").exists():
        paylod = pickle.load(open(smiles_path.parent / f"matcher_{smiles_path.stem}.pkl", "rb"))
    else:
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
                xyz_out = _xtb_optimize(
                    xyz, finaldir, threads=args.num_threads, charge=charge
                )
                if xyz_out is not None:
                    payload[f"stgg{idx}"]["xtb_coordinates_path"] = (
                        xyz_out.absolute().as_posix()
                    )
            pbar.update(1)

    print(
        f"Finished generating molecule coordinates using XTB. You'll find them in {workdir} !"
    )

    with open(smiles_path.parent / f"matcher_{smiles_path.stem}.pkl", "wb") as f:
        pickle.dump(payload, f)
