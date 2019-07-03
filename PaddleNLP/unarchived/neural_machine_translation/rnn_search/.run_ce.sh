#!/bin/bash


DATA_PATH=./data/en-vi/

train(){
python train.py \
        --src_lang en --tar_lang vi \
        --attention True \
        --num_layers 2 \
        --hidden_size 512 \
        --src_vocab_size 17191 \
        --tar_vocab_size 7709 \
        --batch_size 128 \
        --dropout 0.2 \
        --init_scale  0.1 \
        --max_grad_norm 5.0 \
        --train_data_prefix ${DATA_PATH}/train \
        --eval_data_prefix ${DATA_PATH}/tst2012 \
        --test_data_prefix ${DATA_PATH}/tst2013 \
        --vocab_prefix ${DATA_PATH}/vocab \
        --use_gpu True \
        --max_epoch 2 \
        --enable_ce
}


cudaid=${transformer:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train | python _ce.py

#cudaid=${transformer_m:=0,1,2,3} # use 0,1,2,3 card as default
#export CUDA_VISIBLE_DEVICES=$cudaid

#train | python _ce.py
