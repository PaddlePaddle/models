#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${face_detection:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python -u train.py --config_path 'data/config.txt' --train_dir 'data/paddle_train.txt' --batch_size 32 --epoch_num 1 --use_cuda 1 --enable_ce --batch_num 10000 |  python _ce.py


cudaid=${face_detection_4:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python -u train.py --config_path 'data/config.txt' --train_dir 'data/paddle_train.txt' --batch_size 32 --epoch_num 1 --use_cuda 1 --enable_ce --batch_num 10000 |  python _ce.py


