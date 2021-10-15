#!/bin/sh

DATA_DIR=data/
MODEL_DIR=billsum_model/

# train
python src/main.py \
 --mode train \
 --train_path $DATA_DIR/billsum_train.pkl \
 --val_path $DATA_DIR/billsum_val.pkl \
 --save_dir $MODEL_DIR \
 --num_epochs 12 \
 --early_stop_patience 8 \
 --batch_size 8 \
 --warmup_steps 2000 \
 --eval_steps 1000

# eval baselines
python src/experiments/evaluate_baselines.py --data_path $DATA_DIR/billsum_test.pkl

# eval simcls
python src/main.py \
 --mode test \
 --test_path $DATA_DIR/billsum_test.pkl \
 --model_path $MODEL_DIR