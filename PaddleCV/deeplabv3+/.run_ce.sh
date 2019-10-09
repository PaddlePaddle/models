#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

DATASET_PATH=${HOME}/.cache/paddle/dataset/cityscape/

cudaid=${deeplabv3plus:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py \
--batch_size=2 \
--train_crop_size=769 \
--total_step=50 \
--save_weights_path=output1 \
--dataset_path=$DATASET_PATH \
--enable_ce | python _ce.py

cudaid=${deeplabv3plus_m:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py \
--batch_size=8 \
--train_crop_size=769 \
--total_step=50 \
--save_weights_path=output4 \
--dataset_path=$DATASET_PATH \
--enable_ce | python _ce.py
