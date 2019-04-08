#!/bin/bash

# This file is only used for continuous evaluation.
export FLAGS_cudnn_deterministic=True
BATCH_SIZE=56
cudaid=${object_detection_cudaid:=0}
export CUDA_VISIBLE_DEVICES=$cudaid
python train.py --batch_size=${BATCH_SIZE} --num_epochs=5 --enable_ce=True --lr_strategy=cosine_decay | python _ce.py

BATCH_SIZE=224
cudaid=${object_detection_cudaid_m:=0, 1, 2, 3}
export CUDA_VISIBLE_DEVICES=$cudaid
python train.py --batch_size=${BATCH_SIZE} --num_epochs=5 --enable_ce=True --lr_strategy=cosine_decay | python _ce.py
