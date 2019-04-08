#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


export CPU_NUM=1
export NUM_THREADS=1

FLAGS_benchmark=true  python train.py --train_dir train_big_data --vocab_path vocab_big.txt --use_cuda 0 --batch_size 500 --model_dir model_output --epochs 2 --enable_ce --step_num 500 | python _ce.py


cudaid=${ssr:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --train_dir train_big_data --vocab_path vocab_big.txt --use_cuda 1 --batch_size 500 --model_dir model_output --epochs 2 --enable_ce --step_num 1000 | python _ce.py


cudaid=${ssr_4:=0,1,2,3} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --train_dir train_big_data --vocab_path vocab_big.txt --use_cuda 1 --parallel 1 --num_devices 2 --batch_size 500 --model_dir model_output --epochs 2 --enable_ce --step_num 1000 | python _ce.py
