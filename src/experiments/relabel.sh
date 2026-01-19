#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10

export PATH=$PATH:$projects/xtb-dist/bin
export XTB4STDAHOME=$projects/xtb4stda
export PATH=$PATH:$XTB4STDAHOME/exe

master_csv=$(ls -td $GENERATOR_CHECKPOINT_DIR/datasets/master_*.csv | head -n1)
master_index="$(basename $master_csv | sed 's/.*_\([^\.]*\).*/\1/')"

# Run property prediction
cd $PROPERTY_PREDICTOR_DIR
source .venv/bin/activate
coordinate_dir=$(ls -td $GENERATOR_CHECKPOINT_DIR/*/ | head -n1)


echo $master_csv,$master_index
uv run predict.py --source_csv_path=$master_csv --use_xtb --checkpoint_path="${PROPERTY_PREDICTOR_DIR}/results/STGGDataset/labeling/1/0/last.ckpt" --xtb_results_folder=$coordinate_dir --output_path=$GENERATOR_CHECKPOINT_DIR/datasets/relabeled_master_${master_index}.csv

deactivate
cd $GENERATOR_DIR
source .venv/bin/activate
cd src
python relabel.py --master_path=$master_csv --relabel_master_path=$GENERATOR_CHECKPOINT_DIR/datasets/relabeled_master_${master_index}.csv

mv $master_csv $GENERATOR_CHECKPOINT_DIR/datasets/old_master_${master_index}.csv
mv $GENERATOR_CHECKPOINT_DIR/datasets/relabeled_master_${master_index}.csv $master_csv
