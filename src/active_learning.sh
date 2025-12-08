#!/bin/bash


# ==== CONFIG ====
N_STEPS=1  # number of AL iterations
n_samples=10000
batch_size=500
temperature_min=0.7
temperature_max=0.7

GENERATOR_DIR=$HOME/projects/rrg-bengioy-ad/jaggbow/STGG-AL
PROPERTY_PREDICTOR_DIR=$HOME/projects/rrg-bengioy-ad/jaggbow/hamiltonian
GENERATOR_CHECKPOINT_DIR=$SCRATCH/AutoregressiveMolecules_checkpoints/jmt_cont_core
GAUSSIAN_DIR=$SCRATCH/AutoregressiveMolecules_checkpoints/gaussian

prev_parse_id=""

for step in $(seq 1 $N_STEPS); do
    echo "============================"
    echo "=== Active Learning Step $step ==="
    echo "============================"

    cd $GENERATOR_DIR/src
    if [ -z "$prev_parse_id" ]; then
        gen_id=$(sbatch experiments/jmt_cont_core.sh \
            --temperature_min $temperature_min \
            --temperature_max $temperature_max \
            --num_samples_ood $n_samples \
            --sample_batch_size $batch_size | awk '{print $4}')
        cd $PROPERTY_PREDICTOR_DIR
        prop_id=$(sbatch train.sh | awk '{print $4}')
        echo "[$step] Property predictor: $prop_id"
    else
        gen_id=$(sbatch --dependency=afterok:$prev_parse_id experiments/jmt_cont_core.sh \
            --temperature_min $temperature_min \
            --temperature_max $temperature_max \
            --num_samples_ood $n_samples \
            --sample_batch_size $batch_size | awk '{print $4}')
        cd $PROPERTY_PREDICTOR_DIR
        prop_id=$(sbatch --dependency=afterok:$prev_parse_id train.sh | awk '{print $4}')
        echo "[$step] Property predictor: $prop_id"

    fi
    echo "[$step] Generator: $gen_id"

    cd $GENERATOR_DIR/src
    coord_id=$(sbatch --dependency=afterok:$gen_id experiments/filter_and_compute_coordinates.sh | awk '{print $4}')
    echo "[$step] Coordinates: $coord_id"

    pp_filter=$(sbatch --dependency=afterok:$prop_id:$coord_id \
        --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,PROPERTY_PREDICTOR_DIR=$PROPERTY_PREDICTOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR \
        experiments/filter_prop.sh | awk '{print $4}')
    echo "[$step] Filter properties: $pp_filter"
  
    while squeue -j $pp_filter > /dev/null 2>&1; do
        sleep 180
    done

    gaussian_id=$(sbatch run_gaussian_array.sh | awk '{print $4}')
    echo "[$step] Gaussian: $gaussian_id"

    parse_id=$(sbatch --dependency=afterany:$gaussian_id \
        --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,PROPERTY_PREDICTOR_DIR=$PROPERTY_PREDICTOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR,GAUSSIAN_DIR=$GAUSSIAN_DIR \
        experiments/parse_results.sh | awk '{print $4}')
    echo "[$step] Parse results: $parse_id"
    
    prev_parse_id=$parse_id
    echo "Waiting for step $step to finish (job $parse_id)..."
    
    while squeue -j $parse_id > /dev/null 2>&1; do
        sleep 180
    done

done

echo "All $N_STEPS steps submitted successfully."

