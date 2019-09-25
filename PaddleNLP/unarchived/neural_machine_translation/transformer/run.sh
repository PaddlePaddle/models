#!/bin/bash

set -ex


#DATA_PATH=$HOME/.cache/paddle/dataset/wmt16

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -u train.py \
      --src_vocab_fpath $DATA_PATH/en_10000.dict \
      --trg_vocab_fpath $DATA_PATH/de_10000.dict \
      --special_token '<s>' '<e>' '<unk>' \
      --train_file_pattern $DATA_PATH/wmt16/train \
      --val_file_pattern $DATA_PATH/wmt16/val \
      --use_token_batch True \
      --batch_size 2048 \
      --sort_type pool \
      --pool_size 10000 \
      --enable_ce True \
      weight_sharing False \
      pass_num 20 \
      dropout_seed 10 \
      --use_iterable_py_reader $@
