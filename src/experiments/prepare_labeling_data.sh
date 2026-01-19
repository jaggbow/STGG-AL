#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10

master_csv=$(ls -td $GENERATOR_CHECKPOINT_DIR/datasets/master_*.csv | head -n1)
master_index="$(basename $master_csv | sed 's/.*_\([^\.]*\).*/\1/')"

cd $GENERATOR_DIR
source .venv/bin/activate
cd src

python prepare_labeling_data.py --master_path=$master_csv
