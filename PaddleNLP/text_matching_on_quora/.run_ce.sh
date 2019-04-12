#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

cudaid=${text_matching_on_quora:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train_and_evaluate.py --model_name=cdssmNet --config=cdssm_base --enable_ce --epoch_num=5 | python _ce.py

cudaid=${text_matching_on_quora_m:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train_and_evaluate.py --model_name=cdssmNet --config=cdssm_base --enable_ce --epoch_num=5 | python _ce.py
