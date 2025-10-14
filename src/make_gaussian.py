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
GJF_DIRECTORY = Path(
    "/network/scratch/o/oussama.boussif/AutoregressiveMolecules_checkpoints/gaussian"
)
SBATCH_DIRECTORY = Path(
    "/network/scratch/o/oussama.boussif/AutoregressiveMolecules_checkpoints/sbatch"
)

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
            + "\n".join(lines)
            + "\n"
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
        with open(gjf_path, "w") as f:
            f.write(gjf_content)

        print(f"[✓] Created {gjf_path}")

    g16_command = "g16"  # Change to full path if needed

    cpus = nproc
    mem = f"{mem}G"
    partition = "rrg-bengioy-ad"  # Change to your cluster’s partition

    for gjf_path in gjf_dir.glob("*.gjf"):
        with open("tmp_script", "w") as f:
            f.write(f"""#!/bin/bash
    #SBATCH --account={partition}
    #SBATCH --job-name={gjf_path.stem}
    #SBATCH --output={SCRATCH_DIR}/slurm-%j.out
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={cpus}
    #SBATCH --mem={mem}
    #SBATCH --time={time}

    module load gaussian/g16 

    export GAUSS_SCRDIR=$SCRATCH/$SLURM_JOB_ID
    mkdir -p $GAUSS_SCRDIR

    g16 {gjf_path}

    rm -rf $GAUSS_SCRDIR
    """)

        sbatch_response = subprocess.check_output(
            ["sbatch tmp_script"], shell=True
        ).decode()  # submit jobs
        print(sbatch_response)
