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

smiles_path=$(ls -t "$GENERATOR_CHECKPOINT_DIR"/*.pkl 2>/dev/null | grep -E '/[0-9]+\.pkl$' | head -n 1)

# Run property prediction
cd $PROPERTY_PREDICTOR_DIR
source .venv/bin/activate
coordinate_dir=$(ls -td $GENERATOR_CHECKPOINT_DIR/*/ | head -n1)

csv_path="${smiles_path%.pkl}_labeling.csv"
echo $coordinate_dir,$smiles_path,$csv_path
uv run predict.py --source_csv_path=$csv_path --use_xtb --checkpoint_path="${PROPERTY_PREDICTOR_DIR}/results/STGGDataset/labeling/1/0/last.ckpt" --xtb_results_folder=$coordinate_dir --output_path=$csv_path

