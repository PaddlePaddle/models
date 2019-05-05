#!/bin/bash

train(){
python  -u run.py   \
        --pass_num 1 \
        --learning_rate 0.001 \
        --batch_size 8 \
        --embed_size 300 \
        --hidden_size 150 \
        --max_p_num 5 \
        --max_p_len 500 \
        --max_q_len 60 \
        --max_a_len 200 \
        --enable_ce \
        --train 
}

cudaid=${single:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train | python _ce.py

cudaid=${multi:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train | python _ce.py
