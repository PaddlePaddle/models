#!/bin/bash

# This file is only used for continuous evaluation.

rm -rf *_factor.txt
python train.py --use_gpu=True 1> log
cat log | python _ce.py
