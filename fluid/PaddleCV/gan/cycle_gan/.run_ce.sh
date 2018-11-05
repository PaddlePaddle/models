#!/bin/bash

# This file is only used for continuous evaluation.
export FLAGS_cudnn_deterministic=True
export ce_mode=1
CUDA_VISIBLE_DEVICES=0 python train.py --batch_size=1 --epoch=10 --run_ce=True --use_gpu=True | python _ce.py


