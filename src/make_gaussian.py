import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

if "SCRATCH" in os.environ:
    SCRATCH_DIR = os.environ["SCRATCH"]
else:
    SCRATCH_DIR = "./"
GJF_DIRECTORY = Path(SCRATCH_DIR)/ "AutoregressiveMolecules_checkpoints/gaussian"
SBATCH_DIRECTORY = Path(SCRATCH_DIR) / "AutoregressiveMolecules_checkpoints/sbatch"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--cpus", default=16, type=int)
    parser.add_argument("--mem", default=16, type=int)
    parser.add_argument("--time", default="8:00:00", type=str)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    gjf_dir = GJF_DIRECTORY / csv_path.stem  # Folder with .gjf files
    smiles = df.SMILES.values
    ids = df.id.values
    xtb_coordinate_paths = df.xtb_coordinates_path.values

    nproc = args.cpus
    mem = args.mem
    time = args.time
    os.makedirs(gjf_dir, exist_ok=True)
    gjf_paths = []
    for id_, xtb_coord_path, smi in tqdm(zip(ids, xtb_coordinate_paths, smiles)):
        with open(xtb_coord_path, "r") as f:
            lines = f.readlines()
            lines = lines[2:]
        title = id_
        # First step: optimization
        gjf_part1 = (
            f"""%mem={mem}GB
%nproc={nproc}
%chk=/scratch/jaggbow/{title}
#p b3lyp/6-31G(d,p) opt scf=(xqc,maxconventionalcycles=100)

{title}

0 1
"""
            + "".join(lines)
        )

        # Second step: TDDFT using optimized geometry
        gjf_part2 = f"""
--Link1--
%mem={mem}GB
%nproc={nproc}
%chk=/scratch/jaggbow/{title}
%nosave
#p b3lyp/6-31G(d,p) geom=check td=(50-50) scf=(xqc,maxconventionalcycles=100)

{title}

0 1
        """

        # Combine and write
        gjf_content = gjf_part1 + gjf_part2
        gjf_path = gjf_dir / f"{title}.gjf"
        gjf_paths.append(gjf_path)
        with open(gjf_path, "w") as f:
            f.write(gjf_content)

        print(f"[✓] Created {gjf_path}")

    g16_command = "g16"  # Change to full path if needed

    cpus = nproc
    mem = f"{mem}G"
    partition = "rrg-bengioy-ad"  # Change to your cluster’s partition
    with open("tmp_gaussian.txt", "w") as f:
        for gjf in gjf_paths:
            f.write(str(gjf) + "\n")

    num_jobs = len(gjf_paths)
    script_path = "run_gaussian_array.sh"
    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --account={partition}
#SBATCH --job-name=gaussian_array
#SBATCH --output={SCRATCH_DIR}/slurm-%A_%a.out
#SBATCH --array=0-{num_jobs-1}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}

module load gaussian/g16

# Pick the corresponding .gjf file for this array index
gjf_file=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" tmp_gaussian.txt)
echo "Running Gaussian on $gjf_file"

export GAUSS_SCRDIR=$SCRATCH/$SLURM_JOB_ID
mkdir -p $GAUSS_SCRDIR

g16 "$gjf_file"

rm -rf $GAUSS_SCRDIR
""")

        sbatch_response = subprocess.check_output(
            ["sbatch", script_path]
        ).decode().strip()  # submit jobs
        print(sbatch_response)
