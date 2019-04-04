#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


export CPU_NUM=1
export NUM_THREADS=1

FLAGS_benchmark=true  python train.py --enable_ce | python _ce.py

