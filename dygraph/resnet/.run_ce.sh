#!/bin/bash

# This file is only used for continuous evaluation.
# dygraph single card
export FLAGS_cudnn_deterministic=True
export CUDA_VISIBLE_DEVICES=0
python train.py --ce --epoch 1 --batch_size 128 | python _ce.py

