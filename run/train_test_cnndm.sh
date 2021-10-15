#!/bin/sh

DATA_DIR=data/
MODEL_DIR=cnndm_model/

# train
python src/main.py \
 --mode train \
 --train_path $DATA_DIR/cnndm_train.pkl \
 --val_path $DATA_DIR/cnndm_val.pkl \
 --save_dir $MODEL_DIR \
 --num_epochs 12 \
 --early_stop_patience 6 \
 --batch_size 20 \
 --warmup_steps 4000 \
 --eval_steps 2000

# eval baselines
python src/experiments/evaluate_baselines.py --data_path $DATA_DIR/cnndm_test.pkl

# eval simcls
python src/main.py \
 --mode test \
 --test_path $DATA_DIR/cnndm_test.pkl \
 --model_path $MODEL_DIR