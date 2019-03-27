#!/bin/bash

#export GLOG_v=30
#export GLOG_logtostderr=1

# start pserver0
export CPU_NUM=5 
export FLAGS_rpc_deadline=3000000
python cluster_train.py \
    --train_data_dir data/convert_text8 \
    --dict_path data/test_build_dict \
    --batch_size 100 \
    --model_output_dir dis_model \
    --base_lr 1.0 \
    --print_batch 1 \
    --is_sparse \
    --with_speed \
    --role pserver \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --current_endpoint 127.0.0.1:6000 \
    --trainers 2 \
    > pserver0.log 2>&1 &

python cluster_train.py \
    --train_data_dir data/convert_text8 \
    --dict_path data/test_build_dict \
    --batch_size 100 \
    --model_output_dir dis_model \
    --base_lr 1.0 \
    --print_batch 1 \
    --is_sparse \
    --with_speed \
    --role pserver \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --current_endpoint 127.0.0.1:6001 \
    --trainers 2 \
    > pserver1.log 2>&1 &

# start trainer0
python cluster_train.py \
    --train_data_dir data/convert_text8 \
    --dict_path data/test_build_dict \
    --batch_size 100 \
    --model_output_dir dis_model \
    --base_lr 1.0 \
    --print_batch 1000 \
    --is_sparse \
    --with_speed \
    --role trainer \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --trainers 2 \
    --trainer_id 0 \
    > trainer0.log 2>&1 &
# start trainer1
python cluster_train.py \
    --train_data_dir data/convert_text8 \
    --dict_path data/test_build_dict \
    --batch_size 100 \
    --model_output_dir dis_model \
    --base_lr 1.0 \
    --print_batch 1000 \
    --is_sparse \
    --with_speed \
    --role trainer \
    --endpoints 127.0.0.1:6000,127.0.0.1:6001 \
    --trainers 2 \
    --trainer_id 1 \
    > trainer1.log 2>&1 &
