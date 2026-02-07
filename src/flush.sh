GENERATOR_CHECKPOINT_DIR=$SCRATCH/AutoregressiveMolecules_checkpoints/test_run
GENERATOR_DIR=$projects/STGG-AL
PROPERTY_PREDICTOR_DIR=$projects/hamiltonian

# Move checkpoints
rm $GENERATOR_CHECKPOINT_DIR/* 

rm $GENERATOR_CHECKPOINT_DIR/checkpoints/*
rm $PROPERTY_PREDICTOR_DIR/results/STGGDataset/filtering/1/0/*
rm $PROPERTY_PREDICTOR_DIR/results/STGGDataset/labeling/1/0/* 

# Move datasets
rm $GENERATOR_DIR/resource/data/jmt_cont_core/data.csv 
rm $GENERATOR_DIR/filtering.csv
rm $GENERATOR_DIR/labeling.csv 

# Delete past checkpoints
find $GENERATOR_CHECKPOINT_DIR -maxdepth 1 -name "*.ckpt" -delete
find $PROPERTY_PREDICTOR_DIR/results -mindepth 1 -name "*.ckpt" -delete

# Delete cache
find $PROPERTY_PREDICTOR_DIR/datasets -mindepth 1 -type f -delete
find $GENERATOR_DIR/resource/data/jmt_cont_core -mindepth 1 -not -name '*.csv' -delete

