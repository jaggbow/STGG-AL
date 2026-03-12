#!/usr/bin/env bash

# ---- AL params ----
N_STEPS=10
TARGET_ORACLE_FREQUENCY=3
n_samples_target_oracle=500

# Generator params
num_samples_ood=5000
sample_batch_size=500
max_epochs=1000
temperature_min=0.7
temperature_max=0.7
tag="main_exp_noisy"

# Property filtering/Labeling params
noise_level=0.3 # same as the master noisy
pp_epochs=100

# ---- Paths ----
GENERATOR_DIR="$HOME/projects/rrg-bengioy-ad/jaggbow/STGG-AL"
PROPERTY_PREDICTOR_DIR="$HOME/projects/rrg-bengioy-ad/jaggbow/hamiltonian"
GENERATOR_CHECKPOINT_DIR="$SCRATCH/AutoregressiveMolecules_checkpoints/$tag"
GAUSSIAN_DIR="$SCRATCH/AutoregressiveMolecules_checkpoints/gaussian"
master_path=$GENERATOR_DIR/master_0_noisy.csv
