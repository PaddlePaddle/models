#!/bin/bash

# This file is only used for continuous evaluation.

export ce_mode=1
rm -rf *_factor.txt
python train.py --use_gpu=True --random_mirror=False --random_scaling=False 1> log
cat log | python _ce.py
