#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --job-name=gaussian_array
#SBATCH --output=/scratch/jaggbow/slurm-%A_%a.out
#SBATCH --array=0-376
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=8:00:00

module load gaussian/g16

# Pick the corresponding .gjf file for this array index
gjf_file=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" tmp_gaussian.txt)
echo "Running Gaussian on $gjf_file"

export GAUSS_SCRDIR=$SCRATCH/$SLURM_JOB_ID
mkdir -p $GAUSS_SCRDIR

g16 "$gjf_file"

rm -rf $GAUSS_SCRDIR
