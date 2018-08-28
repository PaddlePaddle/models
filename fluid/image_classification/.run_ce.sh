#!/bin/bash

# This file is only used for continuous evaluation.
export FLAGS_cudnn_deterministic=True
cudaid=${object_detection_cudaid:=0}
export CUDA_VISIBLE_DEVICES=$cudaid
python train.py --batch_size=64 --num_epochs=5 --enable_ce=True --lr_strategy=cosine_decay | python _ce.py

cudaid=${object_detection_cudaid_m:=0, 1, 2, 3}
export CUDA_VISIBLE_DEVICES=$cudaid
python train.py --batch_size=64 --num_epochs=5 --enable_ce=True --lr_strategy=cosine_decay | python _ce.py
