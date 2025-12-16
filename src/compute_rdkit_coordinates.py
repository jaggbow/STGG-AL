import argparse
import pickle
import subprocess
from pathlib import Path

import ray
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

ray.init()


def _embed_ff_optimize(mol, workdir):
    """
    Return path to the lowest-energy MMFF minimized XYZ for *smiles*.
    The conformer search is embarrassingly parallel – RDKit will multithread.
    """
    charge = Chem.GetFormalCharge(mol)
    xyz = workdir / "basic.xyz"
    if xyz.exists():
        return xyz, charge

    params = AllChem.ETKDGv3()
    params.randomSeed = 0
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol, params) == -1:
        return None, None

    with xyz.open("w") as fh:
        fh.write(f"{mol.GetNumAtoms()}\n\n")
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            fh.write(
                f"{atom.GetSymbol():<3} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n"
            )
    return xyz, charge


@ray.remote(num_cpus=1)
def process_smiles(idx, smi, workdir):
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol, addCoords=True)
        finaldir = workdir / str(idx)
        finaldir.mkdir(exist_ok=True)

        xyz, charge = _embed_ff_optimize(mol, finaldir)
        if xyz is None:
            return None
        return finaldir, xyz, charge, idx, smi
    except Exception as e:
        print(smi, e)
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
                if xyz is not None:
                    payload[f"stgg{idx}"]["basic_coordinates_path"] = (
                        xyz.absolute().as_posix()
                    )
            pbar.update(1)

    print(
        f"Finished generating molecule coordinates using rdkit. You'll find them in {workdir} !"
    )

    with open(smiles_path.parent / f"matcher_{smiles_path.stem}.pkl", "wb") as f:
        pickle.dump(payload, f)
