#!/bin/sh

python src/dataset_builder.py \
 --dataset cnndm \
 --gen_type Bart \
 --gen_path facebook/bart-large-cnn \
 --out_dir data/ \
 --cand_max_len 120 \
 --batch_size 16 \
 --num_cands 16 \
 --num_beam_groups 16 \
 --num_beams 16