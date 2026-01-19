#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10
module load xtb
cd ..
source .venv/bin/activate
cd src

GENERATOR_CHECKPOINT_DIR=$SCRATCH/AutoregressiveMolecules_checkpoints/jmt_cont_core
smiles_path=$(ls -t $GENERATOR_CHECKPOINT_DIR/*.pkl | head -n 1)

python filter_fragments.py --smiles_path=$smiles_path
python compute_xtb_coordinates.py --smiles_path=$smiles_path
