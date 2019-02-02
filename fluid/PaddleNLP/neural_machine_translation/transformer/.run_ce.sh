#!/bin/bash

DATA_PATH=$HOME/.cache/paddle/dataset/wmt16
if [ ! -e $DATA_PATH/en_10000.dict ] ; then
    python -c 'import paddle;paddle.dataset.wmt16.train(10000, 10000, "en")().next()'
    tar -zxf $DATA_PATH/wmt16.tar.gz -C $DATA_PATH
fi

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
        weight_sharing False \
        pass_num 20 \
        dropout_seed 10
}

cudaid=${transformer:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train | python _ce.py

cudaid=${transformer_m:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train | python _ce.py
