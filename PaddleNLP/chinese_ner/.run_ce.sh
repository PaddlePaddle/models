#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${chinese_ner:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --num_passes 300 --device GPU --enable_ce | python _ce.py 

cudaid=${chinese_ner_4:=0,1,2,3} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --num_passes 300 --device GPU --parallel True --enable_ce | python _ce.py 

export CPU_NUM=1
export NUM_THREADS=1

FLAGS_benchmark=true  python train.py --num_passes 300 --device CPU --enable_ce | python _ce.py 
