#!/bin/sh

python src/dataset_builder.py \
 --dataset billsum \
 --gen_type Pegasus \
 --gen_path google/pegasus-billsum \
 --out_dir data/ \
 --cand_max_len 256 \
 --length_penalty 1.25 \
 --diversity_penalty 1.0 \
 --no_repeat_ngram 6 \
 --batch_size 6