#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10

csv_path=$(ls -t "$GENERATOR_CHECKPOINT_DIR"/*.csv 2>/dev/null | grep -E '/[0-9]+\.csv$' | head -n 1)
cd ..
source .venv/bin/activate
cd src

python make_gaussian.py --csv_path=$csv_path
