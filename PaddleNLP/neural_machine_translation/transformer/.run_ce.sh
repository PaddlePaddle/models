#!/bin/bash

sed -i '$a\dropout_seed = 1000' ../../models/neural_machine_translation/transformer/desc.py

DATA_PATH=./dataset/wmt16

train(){
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
        --fetch_steps 1 \
        weight_sharing False \
        pass_num 20
}

cudaid=${transformer:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train | python _ce.py

cudaid=${transformer_m:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train | python _ce.py
