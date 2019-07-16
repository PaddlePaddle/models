#!/bin/bash

# This file is only used for continuous evaluation.
# dygraph single card
export FLAGS_cudnn_deterministic=True
export CUDA_VISIBLE_DEVICES=0
python ptb_dy.py --data_path data/simple-examples/data/ \
               --ce --model_type small | python _ce.py

