#!/bin/bash

set -euo pipefail

# Resolve config relative to script location (important on SLURM)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/$1"

# Make the relevant directories for logging
[ -d $GENERATOR_CHECKPOINT_DIR ] || mkdir $GENERATOR_CHECKPOINT_DIR
[ -d $GENERATOR_CHECKPOINT_DIR/datasets ] || mkdir $GENERATOR_CHECKPOINT_DIR/datasets
[ -d $GENERATOR_CHECKPOINT_DIR/checkpoints ] || mkdir $GENERATOR_CHECKPOINT_DIR/checkpoints

# Copy the master dataset to the logging directory
cp ../master_0.csv $GENERATOR_CHECKPOINT_DIR/datasets

echo "Generator checkpoint directory: $GENERATOR_CHECKPOINT_DIR"
echo "Generator directory: $GENERATOR_DIR"
echo "Property predictor directory: $PROPERTY_PREDICTOR_DIR"

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
cd $PROPERTY_PREDICTOR_DIR
prop_id=$(sbatch --dependency=afterok:$prep_id train_filtering.sh train.epochs=$pp_epochs | awk '{print $4}')
echo "[0] Property predictor: $prop_id"

for step in $(seq 1 $N_STEPS); do
    echo "============================"
    echo "=== Active Learning Step $step ==="
    echo "============================"

    move_labeling_ckpt=false
    move_generator_ckpt=false

    cd $GENERATOR_DIR/src
    filter_prop_id=$(sbatch --dependency=afterok:$prop_id:$gen_id --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,PROPERTY_PREDICTOR_DIR=$PROPERTY_PREDICTOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR experiments/filter_prop.sh | awk '{print $4}')
    echo "[$step] Property filtering: $filter_prop_id"

    xtb_id=$(sbatch --dependency=afterok:$filter_prop_id \
        --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR \
        experiments/run_xtb.sh | awk '{print $4}')
    echo "[$step] XTB computation: $xtb_id"

    continue_id=$xtb_id
    if (( $step % $TARGET_ORACLE_FREQUENCY == 0 )); then
        make_gaussian_id=$(sbatch --dependency=afterok:$continue_id --export=ALL,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR experiments/make_gaussian.sh --n_samples=$n_samples_target_oracle | awk '{print $4}')
        echo "[$step] Make Gaussian: $make_gaussian_id"
        echo "Waiting for step $step to finish (job $make_gaussian_id)..."
        while squeue -j $make_gaussian_id > /dev/null 2>&1; do
            sleep 180
        done
        gaussian_id=$(sbatch run_gaussian_array.sh | awk '{print $4}')
        echo "[$step] Launch Gaussian: $gaussian_id"
        parse_gaussian_id=$(sbatch --dependency=afterany:$gaussian_id --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR,GAUSSIAN_DIR=$GAUSSIAN_DIR experiments/parse_gaussian.sh | awk '{print $4}')
        echo "[$step] Parse Gaussian: $parse_gaussian_id"
        move_labeling_ckpt=true
        move_generator_ckpt=true
        continue_id=$parse_gaussian_id
    fi
    parse_id=$(sbatch --dependency=afterok:$continue_id \
        --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,PROPERTY_PREDICTOR_DIR=$PROPERTY_PREDICTOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR,GAUSSIAN_DIR=$GAUSSIAN_DIR experiments/parse_results.sh $move_labeling_ckpt $move_generator_ckpt | awk '{print $4}')
    echo "[$step] Parse results: $parse_id"

    
    # No labeling model training, just prep the data
    prep_id=$(sbatch --dependency=afterok:$parse_id --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR experiments/prepare_all_data.sh --save_labeling | awk '{print $4}')


    if (( $step < $N_STEPS )); then
        gen_id=$(sbatch --dependency=afterok:$prep_id --export=ALL,GENERATOR_DIR=$GENERATOR_DIR,GENERATOR_CHECKPOINT_DIR=$GENERATOR_CHECKPOINT_DIR experiments/generator.sh \
            --tag $tag \
            --temperature_min $temperature_min \
            --temperature_max $temperature_max \
            --num_samples_ood $num_samples_ood \
            --max_epochs $max_epochs \
            --sample_batch_size $sample_batch_size \
            --seed $step | awk '{print $4}')

        echo "[$step] Generator: $gen_id"
        cd $PROPERTY_PREDICTOR_DIR
        prop_id=$(sbatch --dependency=afterok:$prep_id train_filtering.sh train.epochs=$pp_epochs | awk '{print $4}')
        echo "[$step] Property predictor: $prop_id"
    fi

done
