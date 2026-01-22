#!/bin/bash

set -euo pipefail

# Resolve config relative to script location (important on SLURM)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config_baseline.sh"

[ -d $GENERATOR_CHECKPOINT_DIR ] || mkdir $GENERATOR_CHECKPOINT_DIR
[ -d $GENERATOR_CHECKPOINT_DIR/datasets ] || mkdir $GENERATOR_CHECKPOINT_DIR/datasets
[ -d $GENERATOR_CHECKPOINT_DIR/checkpoints ] || mkdir $GENERATOR_CHECKPOINT_DIR/checkpoints

echo "Generator checkpoint directory: $GENERATOR_CHECKPOINT_DIR"
echo "Generator directory: $GENERATOR_DIR"


# Initial training
prep_id=$(sbatch --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR experiments/prepare_all_data.sh --save_labeling | awk '{print $4}')
echo "[0] Prepare data; $prep_id"
gen_id=$(sbatch --dependency=afterok:$prep_id --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR experiments/generator.sh \
            --tag $tag \
            --temperature_min $temperature_min \
            --temperature_max $temperature_max \
            --num_samples_ood $num_samples_ood \
            --max_epochs $max_epochs \
            --sample_batch_size $sample_batch_size | awk '{print $4}')
echo "[0] Generator: $gen_id"


for step in $(seq 1 $N_STEPS); do
    echo "============================"
    echo "=== Active Learning Step $step ==="
    echo "============================"
   
    cd $GENERATOR_DIR/src

    xtb_id=$(sbatch --dependency=afterok:$gen_id \
        --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR \
        experiments/run_xtb.sh | awk '{print $4}')
    echo "[$step] XTB computation: $xtb_id"
  
    make_gaussian_id=$(sbatch --dependency=afterok:$xtb_id --export=ALL,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR experiments/make_gaussian.sh --n_samples=$n_samples_target_oracle | awk '{print $4}')
    echo "[$step] Make Gaussian: $make_gaussian_id"
    echo "Waiting for step $step to finish (job $make_gaussian_id)..."
    while squeue -j $make_gaussian_id > /dev/null 2>&1; do
    	sleep 180
    done
    gaussian_id=$(sbatch run_gaussian_array.sh | awk '{print $4}')
    echo "[$step] Launch Gaussian: $gaussian_id"
    parse_gaussian_id=$(sbatch --dependency=afterany:$gaussian_id --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR,GAUSSIAN_DIR=$GAUSSIAN_DIR experiments/parse_gaussian.sh | awk '{print $4}')
    echo "[$step] Parse Gaussian: $parse_gaussian_id"
    parse_id=$(sbatch --dependency=afterok:$parse_gaussian_id \
        --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,PROPERTY_PREDICTOR_DIR=$PROPERTY_PREDICTOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR,GAUSSIAN_DIR=$GAUSSIAN_DIR experiments/parse_results.sh $move_filtering_ckpt | awk '{print $4}')
    echo "[$step] Parse results: $parse_id"
    
    prep_id=$(sbatch --dependency=afterok:$parse_id --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR experiments/prepare_all_data.sh --save_labeling | awk '{print $4}')

    gen_id=$(sbatch --dependency=afterok:$prep_id --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR experiments/generator.sh \
            --tag $tag \
            --temperature_min $temperature_min \
            --temperature_max $temperature_max \
            --num_samples_ood $num_samples_ood \
            --max_epochs $max_epochs \
            --sample_batch_size $sample_batch_size | awk '{print $4}')

    echo "[$step] Generator: $gen_id"
done

echo "All $N_STEPS steps submitted successfully."

