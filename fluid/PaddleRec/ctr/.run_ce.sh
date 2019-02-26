#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


#cudaid=${face_detection:=0} # use 0-th card as default
#export CUDA_VISIBLE_DEVICES=$cudaid
export NUM_THREADS=1

FLAGS_benchmark=true  python train.py --is_local 1 --cloud_train 0 --train_data_path data/raw/train.txt --enable_ce | python _ce.py

export NUM_THREADS=4

FLAGS_benchmark=true  python train.py --is_local 1 --cloud_train 0 --train_data_path data/raw/train.txt --enable_ce | python _ce.py

