#!/bin/bash

# This file is only used for continuous evaluation.

python train.py --batch_size=256 --num_passes=10 --total_images=6149 --enable_ce=True | python _ce.py
