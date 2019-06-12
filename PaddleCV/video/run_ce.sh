#!/bin/bash

export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export CUDA_VISIBLE_DEVICES=0

python train.py --model_name="TSM" --config=./configs/tsm.txt --epoch=1 --log_interval=10 --batch_size=128 --enable_ce=True | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py --model_name="TSM" --config=./configs/tsm.txt --epoch=1 --log_interval=10 --batch_size=128 --enable_ce=True | python _ce.py


