import argparse
import os
import pickle
import time
from multiprocessing import Manager, Pool, Process, Queue
from pathlib import Path
from queue import Empty
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


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


def process_smiles_worker(idx, smi, workdir, result_queue):
    """Worker that processes a single SMILES and puts result in queue"""
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol, addCoords=True)
        finaldir = workdir / str(idx)
        finaldir.mkdir(exist_ok=True)
        xyz, charge = _embed_ff_optimize(mol, finaldir)
        if xyz is None:
            result_queue.put((idx, smi, None))
        else:
            result_queue.put((idx, smi, (finaldir, xyz, charge, idx, smi)))
    except Exception as e:
        result_queue.put((idx, smi, ("error", str(e))))


def process_smiles_batch(
    smiles_list, molecule_ids, workdir, n_jobs=None, timeout_seconds=300
):
    """
    Parallelize SMILES processing with hard timeout.

    Args:
        smiles_list: List of tuples (idx, smiles_string)
        workdir: Working directory path
        n_jobs: Number of parallel workers (None = CPU count)
        timeout_seconds: Hard timeout per molecule in seconds

    Returns:
        List of results (None for failed/timeout molecules)
    """
    # Validate n_jobs
    if n_jobs is None:
        n_jobs = os.cpu_count()
    elif n_jobs == -1:
        n_jobs = os.cpu_count()
    elif n_jobs == -2:
        n_jobs = max(1, os.cpu_count() - 1)
    elif n_jobs > os.cpu_count():
        print(
            f"Warning: n_jobs={n_jobs} exceeds CPU count ({os.cpu_count()}). Using {os.cpu_count()} instead."
        )
        n_jobs = os.cpu_count()

    print(
        f"Processing {len(smiles_list)} molecules with {n_jobs} workers (timeout: {timeout_seconds}s each)"
    )

    results = {}
    active_processes = {}
    result_queue = Queue()
    pending_tasks = list(zip(molecule_ids, smiles_list))

    timeout_count = 0
    error_count = 0

    with tqdm(total=len(smiles_list), desc="Processing molecules") as pbar:
        while len(results) < len(smiles_list):
            # Start new processes if slots available and tasks remaining
            while len(active_processes) < n_jobs and pending_tasks:
                idx, smi = pending_tasks.pop(0)
                p = Process(
                    target=process_smiles_worker, args=(idx, smi, workdir, result_queue)
                )
                p.start()
                active_processes[idx] = {
                    "process": p,
                    "smi": smi,
                    "start_time": time.time(),
                }

            # Check for completed processes
            try:
                idx, smi, result = result_queue.get(timeout=0.1)

                if idx in active_processes:
                    active_processes[idx]["process"].join(timeout=1)
                    del active_processes[idx]

                if isinstance(result, tuple) and result[0] == "error":
                    tqdm.write(f"❌ Error: {smi} (idx: {idx}): {result[1]}")
                    results[idx] = None
                    error_count += 1
                else:
                    results[idx] = result

                pbar.update(1)

            except Empty:
                pass

            # Check for timeouts
            current_time = time.time()
            for idx, proc_info in list(active_processes.items()):
                elapsed = current_time - proc_info["start_time"]
                if elapsed > timeout_seconds:
                    tqdm.write(
                        f"⏱ Timeout ({timeout_seconds}s): {proc_info['smi']} (idx: {idx})"
                    )
                    proc_info["process"].terminate()
                    proc_info["process"].join(timeout=2)
                    if proc_info["process"].is_alive():
                        proc_info["process"].kill()
                        proc_info["process"].join()

                    del active_processes[idx]
                    results[idx] = None
                    timeout_count += 1
                    pbar.update(1)

            # Small sleep to prevent busy waiting
            if not result_queue.empty() or not active_processes:
                continue
            time.sleep(0.05)

    # Clean up any remaining processes
    for proc_info in active_processes.values():
        if proc_info["process"].is_alive():
            proc_info["process"].terminate()
            proc_info["process"].join()

    # Convert dict to ordered list
    ordered_results = [results.get(idx, None) for idx in molecule_ids]

    successful = sum(1 for r in ordered_results if r is not None)
    print(
        f"\n✓ Successful: {successful}/{len(smiles_list)} | ⏱ Timeouts: {timeout_count} | ❌ Errors: {error_count}"
    )

    return ordered_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type=str)
    parser.add_argument("--timeout", type=int, default=75)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    smiles_path = Path(args.smiles_path)
    data = pickle.load(open(smiles_path, "rb"))
    smiles_list = data["smiles"]
    molecule_ids = data["molecule_id"]

    workdir = smiles_path.parent / smiles_path.stem
    workdir.mkdir(exist_ok=True, parents=True)
    
    results = process_smiles_batch(
        smiles_list,
        molecule_ids,
        workdir,
        n_jobs=args.num_workers,  # use all CPUs
        timeout_seconds=args.timeout,
    )
    
    if (smiles_path.parent / f"filtering_{smiles_path.stem}.pkl").exists():
        payload = pickle.load(open(smiles_path.parent / f"filtering_{smiles_path.stem}.pkl", "rb"))
    else:
        payload = {}
    
    for result in results:
        if result is not None:
            finaldir, xyz, charge, mol_id, smi = result
            if xyz is not None:
                payload[mol_id] = {"SMILES": smi}
                payload[mol_id]["basic_coordinates_path"] = xyz.absolute().as_posix()
    print(
        f"Finished generating molecule coordinates using rdkit. You'll find them in {workdir} !"
    )

    with open(smiles_path.parent / f"filtering_{smiles_path.stem}.pkl", "wb") as f:
        pickle.dump(payload, f)
    df_dict = {
        "molecule_id": [k for k in payload],
        "SMILES": [payload[k]["SMILES"] for k in payload],
        "basic_coordinates_path": [payload[k]["basic_coordinates_path"] for k in payload]
    }
    df_to_labeling = pd.DataFrame.from_dict(df_dict)
    df_to_labeling.to_csv(smiles_path.parent / f"{smiles_path.stem}_filtering.csv")

