#!/bin/bash

# This file is only used for continuous evaluation.
# dygraph single card
export FLAGS_cudnn_deterministic=True
export CUDA_VISIBLE_DEVICES=0
python main.py --ce --epoch 1 --random_seed 33 --validation_steps 600 | python _ce.py

