#!/bin/bash

#export GLOG_v=30
#export GLOG_logtostderr=1

# start pserver0
python cluster_train.py \
    --train_dir train_data \
    --model_dir cluster_model \
    --vocab_path vocab.txt \
    --batch_size 5 \
    --is_local 0 \
    --role pserver \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --current_endpoint 127.0.0.1:6000 \
    --trainers 2 \
    > pserver0.log 2>&1 &

# start pserver1
python cluster_train.py \
    --train_dir train_data \
    --model_dir cluster_model \
    --vocab_path vocab.txt \
    --batch_size 5 \
    --is_local 0 \
    --role pserver \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --current_endpoint 127.0.0.1:6001 \
    --trainers 2 \
    > pserver1.log 2>&1 &

# start trainer0
#CUDA_VISIBLE_DEVICES=1 python cluster_train.py \
python cluster_train.py \
    --train_dir train_data \
    --model_dir cluster_model \
    --vocab_path vocab.txt \
    --batch_size 5 \
    --print_batch 10 \
    --use_cuda 0 \
    --is_local 0 \
    --role trainer \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --trainers 2 \
    --trainer_id 0 \
    > trainer0.log 2>&1 &

# start trainer1
#CUDA_VISIBLE_DEVICES=2 python cluster_train.py \
python cluster_train.py \
    --train_dir train_data \
    --model_dir cluster_model \
    --vocab_path vocab.txt \
    --batch_size 5 \
    --print_batch 10 \
    --use_cuda 0 \
    --is_local 0 \
    --role trainer \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --trainers 2 \
    --trainer_id 1 \
    > trainer1.log 2>&1 &
