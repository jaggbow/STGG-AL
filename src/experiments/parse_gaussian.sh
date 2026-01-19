#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10

smiles_path=$(ls -t "$GENERATOR_CHECKPOINT_DIR"/*.pkl 2>/dev/null | grep -E '/[0-9]+\.pkl$' | head -n 1)
csv_path="${smiles_path%.pkl}_labeling.csv"
filename=$(basename "$csv_path")
stem="${filename%.*}"
gaussian_dir=$GAUSSIAN_DIR/$stem
cd $GENERATOR_DIR
source .venv/bin/activate
cd src

python parse_gaussian.py --smiles_path=$smiles_path --labeling_csv_path=$csv_path --gaussian_dir=$gaussian_dir
