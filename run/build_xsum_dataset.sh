#!/bin/sh

python src/dataset_builder.py \
 --dataset xsum \
 --gen_type Pegasus \
 --gen_path google/pegasus-xsum \
 --out_dir data/ \
 --cand_max_len 80 \
 --batch_size 16 \
 --num_cands 4 \
 --num_beam_groups 4 \
 --num_beams 4