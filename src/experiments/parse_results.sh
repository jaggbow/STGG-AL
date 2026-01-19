#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=50:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10
module load xtb

move_filtering_ckpt=$1

smiles_path=$(ls -t "$GENERATOR_CHECKPOINT_DIR"/*.pkl 2>/dev/null | grep -E '/[0-9]+\.pkl$' | head -n 1)
old_master_csv=$(ls -td $GENERATOR_CHECKPOINT_DIR/datasets/master_*.csv | head -n1)
old_master_index="$(basename $old_master_csv | sed 's/.*_\([^\.]*\).*/\1/')"

cd $GENERATOR_DIR
source .venv/bin/activate
cd src
python labeling.py --smiles_path=$smiles_path --old_master_path=$old_master_csv

# Move checkpoints
mv $GENERATOR_CHECKPOINT_DIR/last.ckpt $GENERATOR_CHECKPOINT_DIR/checkpoints/generator_${old_master_index}.ckpt
mv $PROPERTY_PREDICTOR_DIR/results/STGGDataset/filtering/1/0/last.ckpt $GENERATOR_CHECKPOINT_DIR/checkpoints/filtering_${old_master_index}.ckpt
if [ "$move_filtering_ckpt" = true ] ; then
    mv $PROPERTY_PREDICTOR_DIR/results/STGGDataset/labeling/1/0/last.ckpt $GENERATOR_CHECKPOINT_DIR/checkpoints/labeling_${old_master_index}.ckpt
fi

# Move datasets
mv $GENERATOR_DIR/resource/data/jmt_cont_core/data.csv $GENERATOR_CHECKPOINT_DIR/datasets/generator_${old_master_index}.csv
mv $GENERATOR_DIR/filtering.csv $GENERATOR_CHECKPOINT_DIR/datasets/filtering_${old_master_index}.csv
mv $GENERATOR_DIR/labeling.csv $GENERATOR_CHECKPOINT_DIR/datasets/labeling_${old_master_index}.csv

# Delete past checkpoints
find $GENERATOR_CHECKPOINT_DIR -maxdepth 1 -name "*.ckpt" -delete
find $PROPERTY_PREDICTOR_DIR/results -mindepth 1 -name "*.ckpt" -delete

# Delete cache
find $PROPERTY_PREDICTOR_DIR/datasets -mindepth 1 -type f -delete
find $GENERATOR_DIR/resource/data/jmt_cont_core -mindepth 1 -not -name '*.csv' -delete
