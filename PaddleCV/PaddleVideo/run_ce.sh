#!/bin/bash

export FLAGS_fraction_of_gpu_memory_to_use=0.98
export CUDA_VISIBLE_DEVICES=0

python train.py --model_name="TSM" --config=./configs/tsm.txt --epoch=1 --log_interval=10 --batch_size=128 --enable_ce=True | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py --model_name="TSM" --config=./configs/tsm.txt --epoch=1 --log_interval=10 --batch_size=128 --enable_ce=True | python _ce.py


