#!/bin/sh

DATA_DIR=data/
MODEL_DIR=cnndm_model/

# number of test candidates vs performance
python src/experiments/evaluate_num_candidates.py \
 --model_path $MODEL_DIR \
 --test_path $DATA_DIR/cnndm_test.pkl \
 --out_dir figures/ \

# build the data to run visualization and extra experiments
python src/experiments/build_experiment_data.py \
 --model_path $MODEL_DIR \
 --test_path $DATA_DIR/cnndm_test.pkl \
 --out_path $DATA_DIR/experiment_data.pkl \

# entity-level evaluation
python src/experiments/entity_level_eval.py \
 --data_path $DATA_DIR/experiment_data.pkl

 # sentence-level evaluation
python src/experiments/sentence_level_eval.py \
 --data_path $DATA_DIR/experiment_data.pkl \
 --out_dir figures/


