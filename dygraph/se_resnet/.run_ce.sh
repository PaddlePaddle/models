#!/bin/bash

# This file is only used for continuous evaluation.
# dygraph single card
export FLAGS_cudnn_deterministic=True
export CUDA_VISIBLE_DEVICES=5
python -u train.py --ce --epoch 1 | python _ce.py
#python train.py --ce --epoch 1 | python _ce.py

