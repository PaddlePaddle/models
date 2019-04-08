#!/bin/bash

#export GLOG_v=30
#export GLOG_logtostderr=1

python -u cluster_train.py \
--config_path 'data/config.txt' \
--train_dir 'data/paddle_train.txt' \
--batch_size 32 \
--epoch_num 100 \
--use_cuda 0 \
--parallel 0 \
--role pserver \
--endpoints 127.0.0.1:6000,127.0.0.1:6001 \
--current_endpoint 127.0.0.1:6000 \
--trainers 2 \
> pserver0.log 2>&1 &

python -u cluster_train.py \
--config_path 'data/config.txt' \
--train_dir 'data/paddle_train.txt' \
--batch_size 32 \
--epoch_num 100 \
--use_cuda 0 \
--parallel 0 \
--role pserver \
--endpoints 127.0.0.1:6000,127.0.0.1:6001 \
--current_endpoint 127.0.0.1:6001 \
--trainers 2 \
> pserver1.log 2>&1 &

python -u cluster_train.py \
--config_path 'data/config.txt' \
--train_dir 'data/paddle_train.txt' \
--batch_size 32 \
--epoch_num 100 \
--use_cuda 0 \
--parallel 0 \
--role trainer \
--endpoints 127.0.0.1:6000,127.0.0.1:6001 \
--trainers 2 \
--trainer_id 0 \
> trainer0.log 2>&1 &

python -u cluster_train.py \
--config_path 'data/config.txt' \
--train_dir 'data/paddle_train.txt' \
--batch_size 32 \
--epoch_num 100 \
--use_cuda 0 \
--parallel 0 \
--role trainer \
--endpoints 127.0.0.1:6000,127.0.0.1:6001 \
--trainers 2 \
--trainer_id 1 \
> trainer1.log 2>&1 &
