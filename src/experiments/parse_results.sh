#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=30:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10
module load xtb

coordinate_dir=$(ls -td $GENERATOR_CHECKPOINT_DIR/*/ | head -n1)
smiles_path=$(ls -t "$GENERATOR_CHECKPOINT_DIR"/*.pkl 2>/dev/null | grep -E '/[0-9]+\.pkl$' | head -n 1)
csv_path="${smiles_path%.pkl}.csv"
filename=$(basename $coordinate_dir)
gaussian_path=$GAUSSIAN_DIR/$filename

cd $GENERATOR_DIR
source .venv/bin/activate
cd src
python parse_results.py --gaussian_dir=$gaussian_path --csv_path=$csv_path --coordinates_dir=$coordinate_dir --generator_data_path=$GENERATOR_DIR/resource/data/jmt_cont_core/data.csv --property_predictor_data_path=/scratch/jaggbow/jmt.csv
# Delete past checkpoints
find $GENERATOR_CHECKPOINT_DIR -maxdepth 1 -name "*.ckpt" -delete
find $PROPERTY_PREDICTOR_DIR/results -mindepth 1 -name "*.ckpt" -delete

# Delete cache
find $PROPERTY_PREDICTOR_DIR/datasets -mindepth 1 -type f -delete
find $GENERATOR_DIR/resource/data/jmt_cont_core -mindepth 1 -not -name '*.csv' -delete
