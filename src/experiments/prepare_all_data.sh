#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10

master_csv=$(ls -td $GENERATOR_CHECKPOINT_DIR/datasets/master_*.csv | head -n1)

cd $GENERATOR_DIR
source .venv/bin/activate
cd src

python prepare_data.py --master_path=$master_csv --generator_data_dir=$GENERATOR_DIR/resource/data/jmt_cont_core "$@"

# Push data
#git add ../resource/data/jmt_cont_core/data.csv
#git commit -m "updated generator data"
#git push

