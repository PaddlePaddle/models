#!/bin/bash

# This file is only used for continuous evaluation.
cudaid=${object_detection_cudaid:=0}
export CUDA_VISIBLE_DEVICES=$cudaid
python train.py --batch_size=64 --num_passes=10 --total_images=6149 --enable_ce=True | python _ce.py

cudaid=${object_detection_cudaid:=0, 1, 2, 3}
export CUDA_VISIBLE_DEVICES=$cudaid
python train.py --batch_size=64 --num_passes=10 --total_images=6149 --enable_ce=True | python _ce.py
