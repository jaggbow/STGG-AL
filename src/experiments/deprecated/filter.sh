#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10
module load cuda/12.2
module load xtb
cd ..
deactivate
source .venv/bin/activate
cd src

python filter.py --smiles_path /scratch/jaggbow/AutoregressiveMolecules_checkpoints/jmt_cont_core/1757847988568.pkl --property_predictor_ckpt_path /home/jaggbow/projects/rrg-bengioy-ad/jaggbow/hamiltonian/results/STGGDataset/test/1/0/last-v1.ckpt --property_predictor_cwd /home/jaggbow/projects/rrg-bengioy-ad/jaggbow/hamiltonian
