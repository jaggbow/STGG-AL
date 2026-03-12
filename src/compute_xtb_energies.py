import argparse
import subprocess
from pathlib import Path
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from multiprocessing import Process, Queue
import os
import time
from queue import Empty
import hashlib
import numpy as np


VS1_XTB_RMS = 3.298644208969044
VDELTA_XTB_RMS = 0.81681177503766


def smiles_seeded_noise(smiles, sigma):
    seed = int(hashlib.md5(smiles.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    return rng.randn() * sigma

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
        try:
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
        except Chem.KekulizeException as e:  # noqa: F841
            pass

    if len(energies) > 0:
        best_cid, best_e = min(energies, key=lambda x: x[1])
    else:
        best_cid, best_e = ids[0], None

    with xyz.open("w") as fh:
        if best_e is None:
            fh.write(f"{mol.GetNumAtoms()}\n\n")
        else:
            fh.write(f"{mol.GetNumAtoms()}\nMMFF (kcal mol-1): {best_e:.3f}\n")
        conf = mol.GetConformer(best_cid)
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            fh.write(
                f"{atom.GetSymbol():<3} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n"
            )
    return xyz, charge


def process_smiles_worker(mol_id, smi, workdir, result_queue):
    """Worker that processes a single SMILES and puts result in queue"""
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol, addCoords=True)
        finaldir = workdir / str(mol_id)
        finaldir.mkdir(exist_ok=True)

        xyz, charge = _embed_ff_optimize(mol, finaldir)
        if xyz is None:
            result_queue.put((mol_id, smi, None))
        else:
            result_queue.put((mol_id, smi, (finaldir, xyz, charge, mol_id, smi)))
    except Exception as e:
        print(smi, e)
        result_queue.put((mol_id, smi, None))


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

    with (workdir / "xtb.log").open("w") as log:
        subprocess.run(
            cmd, cwd=workdir, check=True, stdout=log, stderr=subprocess.STDOUT
        )

    if not xyz_out.exists():
        raise RuntimeError("xTB did not produce geom.xtbopt.xyz")
    return xyz_out


def _xtb_energies(xyz_out: Path, workdir: Path, stda_cutoff: int = 10):
    cmd = "xtb4stda {} >& xtb4stda.out".format("geom.xtbopt.xyz")
    subprocess.run(
        cmd, cwd=workdir, shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL
    )
    # Run singlet run
    cmd = f"stda_v1.6.3 -xtb -e {stda_cutoff} >& stda_xtb_singlet.out"
    subprocess.run(
        cmd, cwd=workdir, shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL
    )
    with open(workdir / "tda.dat") as f:
        for line in f:
            if line.startswith(" DATXY"):
                break
        columns = ["transition", "energy", "fosc", "rot_x", "rot_y", "rot_z"]
        df = pd.read_csv(f, header=None, names=columns, sep="\s+")
        vs1 = df.iloc[0]["energy"]

    # Run triplet run
    cmd = f"stda_v1.6.3 -xtb -t -e {stda_cutoff} >& stda_xtb_triplet.out"
    subprocess.run(
        cmd, cwd=workdir, shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL
    )
    with open(workdir / "tda.dat") as f:
        for line in f:
            if line.startswith(" DATXY"):
                break
        columns = ["transition", "energy", "fosc", "rot_x", "rot_y", "rot_z"]
        df = pd.read_csv(f, header=None, names=columns, sep="\s+")
        vt1 = df.iloc[0]["energy"]
    return vs1, vt1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type=str)
    parser.add_argument("--timeout", type=int, default=75)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--noise_level", type=float, default=0)
    args = parser.parse_args()

    smiles_path = Path(args.smiles_path)
    data = pickle.load(open(smiles_path, "rb"))
    smiles_list = data["smiles"]
    molecule_id = data["molecule_id"]

    workdir = smiles_path.parent / smiles_path.stem
    workdir.mkdir(exist_ok=True, parents=True)

    # Determine number of workers
    n_jobs = args.num_workers
    timeout_seconds = args.timeout
    
    print(f"Processing {len(smiles_list)} molecules with {n_jobs} workers (timeout: {timeout_seconds}s each)")

    # Load existing payload if available
    if (smiles_path.parent / f"labeling_{smiles_path.stem}.pkl").exists():
        payload = pickle.load(open(smiles_path.parent / f"labeling_{smiles_path.stem}.pkl", "rb"))
    else:
        payload = {}

    # Prepare task list
    tasks = list(zip(molecule_id, smiles_list))
    results = {}
    active_processes = {}
    result_queue = Queue()
    pending_tasks = tasks.copy()
    
    timeout_count = 0
    error_count = 0

    with tqdm(total=len(tasks), desc="Processing molecules") as pbar:
        while len(results) < len(tasks):
            # Start new processes if slots available and tasks remaining
            while len(active_processes) < n_jobs and pending_tasks:
                mol_id, smi = pending_tasks.pop(0)
                p = Process(target=process_smiles_worker, args=(mol_id, smi, workdir, result_queue))
                p.start()
                active_processes[mol_id] = {
                    'process': p,
                    'smi': smi,
                    'start_time': time.time()
                }

            # Check for completed processes
            try:
                mol_id, smi, result = result_queue.get(timeout=0.1)
                
                if mol_id in active_processes:
                    active_processes[mol_id]['process'].join(timeout=1)
                    del active_processes[mol_id]

                if result is not None:
                    finaldir, xyz, charge, mol_id, smi = result
                    try:
                        xyz_out = _xtb_optimize(xyz, finaldir, charge=charge)
                        vs1, vt1 = _xtb_energies(xyz_out, finaldir, stda_cutoff=10)
                        payload[mol_id] = {"SMILES": smi}
                        payload[mol_id]["xtb_coordinates_path"] = xyz_out.absolute().as_posix()
                        payload[mol_id]["vs1_xtb"] = vs1 + smiles_seeded_noise(smi, args.noise_level * VS1_XTB_RMS)
                        payload[mol_id]["vdelta_xtb"] = (vs1 - vt1) + smiles_seeded_noise(smi, args.noise_level * VDELTA_XTB_RMS)
                    except Exception as e:
                        tqdm.write(f"❌ XTB Error for {smi} (id: {mol_id}): {e}")
                        error_count += 1
                else:
                    error_count += 1

                results[mol_id] = result
                pbar.update(1)

            except Empty:
                pass

            # Check for timeouts
            current_time = time.time()
            for mol_id, proc_info in list(active_processes.items()):
                elapsed = current_time - proc_info['start_time']
                if elapsed > timeout_seconds:
                    tqdm.write(f"⏱ Timeout ({timeout_seconds}s): {proc_info['smi']} (id: {mol_id})")
                    proc_info['process'].terminate()
                    proc_info['process'].join(timeout=2)
                    if proc_info['process'].is_alive():
                        proc_info['process'].kill()
                        proc_info['process'].join()
                    
                    del active_processes[mol_id]
                    results[mol_id] = None
                    timeout_count += 1
                    pbar.update(1)

            # Small sleep to prevent busy waiting
            if not result_queue.empty() or not active_processes:
                continue
            time.sleep(0.05)

    # Clean up any remaining processes
    for proc_info in active_processes.values():
        if proc_info['process'].is_alive():
            proc_info['process'].terminate()
            proc_info['process'].join()

    successful = len(payload)
    print(f"\n✓ Successful: {successful}/{len(tasks)} | ⏱ Timeouts: {timeout_count} | ❌ Errors: {error_count}")
    print(f"Finished generating molecule coordinates using XTB. You'll find them in {workdir}!")

    with open(smiles_path.parent / f"labeling_{smiles_path.stem}.pkl", "wb") as f:
        pickle.dump(payload, f)
    
    df_dict = {
        "molecule_id": [k for k in payload],
        "SMILES": [payload[k]["SMILES"] for k in payload],
        "xtb_coordinates_path": [payload[k]["xtb_coordinates_path"] for k in payload],
        "vs1_xtb": [payload[k]["vs1_xtb"] for k in payload],
        "vdelta_xtb": [payload[k]["vdelta_xtb"] for k in payload],
    }
    df_to_labeling = pd.DataFrame.from_dict(df_dict)
    df_to_labeling.to_csv(smiles_path.parent / f"{smiles_path.stem}_labeling.csv")
