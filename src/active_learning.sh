#!/bin/bash

GENERATOR_DIR=$HOME/projects/rrg-bengioy-ad/jaggbow/STGG-AL
PROPERTY_PREDICTOR_DIR=$HOME/projects/rrg-bengioy-ad/jaggbow/hamiltonian
GENERATOR_CHECKPOINT_DIR=$SCRATCH/AutoregressiveMolecules_checkpoints/jmt_cont_core
n_samples=10000
batch_size=500
temperature_min=0.7
temperature_max=0.7


cd $GENERATOR_DIR/src
gen_id=$(sbatch experiments/jmt_cont_core.sh --temperature_min $temperature_min --temperature_max $temperature_max --num_samples_ood $n_samples --sample_batch_size $batch_size | awk '{print $4}')
echo "Launched generator training ${gen_id}"

cd $PROPERTY_PREDICTOR_DIR
prop_id=$(sbatch train.sh | awk '{print $4}')
echo "Launched property predictor training ${prop_id}"

cd $GENERATOR_DIR/src
coord_id=$(sbatch --dependency=afterok:$gen_id experiments/filter_and_compute_coordinates.sh | awk '{print $4}')
echo "Computing xtb coordinates and filtering fragments ${coord_id}"

pp_filter=$(sbatch --dependency=afterok:$prop_id:$coord_id --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,PROPERTY_PREDICTOR_DIR=$PROPERTY_PREDICTOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR experiments/filter_prop.sh | awk '{print $4}')
echo "Predicting properties and filtering ${pp_filter}"

