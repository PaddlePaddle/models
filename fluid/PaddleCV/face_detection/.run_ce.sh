#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${face_detection:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --batch_size=2 --epoc_num=1 --batch_num=200 --parallel=False --enable_ce | python _ce.py


cudaid=${face_detection_m:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --batch_size=8 --epoc_num=1 --batch_num=200 --parallel=False --enable_ce | python _ce.py

