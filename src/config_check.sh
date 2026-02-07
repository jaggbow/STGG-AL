#!/usr/bin/env bash

# ---- AL params ----
N_STEPS=6
TARGET_ORACLE_FREQUENCY=2
n_samples_target_oracle=500
# Generator params
num_samples_ood=5000
sample_batch_size=500
max_epochs=500
temperature_min=0.7
temperature_max=0.7
tag="checker"

# Property filtering/Labeling params
pp_epochs=100

# ---- Paths ----
GENERATOR_DIR="$HOME/projects/rrg-bengioy-ad/jaggbow/STGG-AL"
PROPERTY_PREDICTOR_DIR="$HOME/projects/rrg-bengioy-ad/jaggbow/hamiltonian"
GENERATOR_CHECKPOINT_DIR="$SCRATCH/AutoregressiveMolecules_checkpoints/$tag"
GAUSSIAN_DIR="$SCRATCH/AutoregressiveMolecules_checkpoints/gaussian"

