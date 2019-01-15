#!/bin/bash

# This file is only used for continuous evaluation.
export FLAGS_cudnn_deterministic=True
export ce_mode=1
(CUDA_VISIBLE_DEVICES=2 python c_gan.py --batch_size=121 --epoch=1 --run_ce=True --use_gpu=True & \
CUDA_VISIBLE_DEVICES=3 python dc_gan.py --batch_size=121 --epoch=1 --run_ce=True --use_gpu=True) | python _ce.py


