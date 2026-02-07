#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=24
#SBATCH --mem=72G
#SBATCH --time=10:00:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10

export PATH=$PATH:$projects/xtb-dist/bin
export XTB4STDAHOME=$projects/xtb4stda
export PATH=$PATH:$XTB4STDAHOME/exe

smiles_path=$(ls -t "$GENERATOR_CHECKPOINT_DIR"/*.pkl 2>/dev/null | grep -E '/[0-9]+\.pkl$' | head -n 1)

# Run xtb energy computation
cd $GENERATOR_DIR
source .venv/bin/activate
cd src
echo "Computing xtb energies on ${smiles_path}."
python compute_xtb_energies.py --smiles_path=$smiles_path --num_workers=24 --timeout=180
deactivate

