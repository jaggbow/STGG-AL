#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10
module load xtb

smiles_path=$(ls -t "$GENERATOR_CHECKPOINT_DIR"/*.pkl 2>/dev/null | grep -E '/[0-9]+\.pkl$' | head -n 1)


# Run rdkit coordinate computation
cd $GENERATOR_DIR
source .venv/bin/activate
cd src
python compute_rdkit_coordinates.py --smiles_path=$smiles_path --num_workers=6
deactivate

# Run property prediction
cd $PROPERTY_PREDICTOR_DIR
coordinate_dir=$(ls -td $GENERATOR_CHECKPOINT_DIR/*/ | head -n1)
smiles_path=$(ls -t "$GENERATOR_CHECKPOINT_DIR"/*.pkl 2>/dev/null | grep -E '/[0-9]+\.pkl$' | head -n 1)
csv_path="${smiles_path%.pkl}_filtering.csv"
echo $coordinate_dir,$smiles_path,$csv_path
uv run predict.py --source_csv_path=$csv_path --checkpoint_path="${PROPERTY_PREDICTOR_DIR}/results/STGGDataset/filtering/1/0/last.ckpt" --xtb_results_folder=$coordinate_dir --output_path=$csv_path
deactivate

# Filtering
cd $GENERATOR_DIR
source .venv/bin/activate
cd src
python filter_prop_predictor.py --smiles_path=$smiles_path
