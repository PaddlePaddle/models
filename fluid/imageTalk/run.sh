#!/usr/bin/env sh

#you can use 'python train.py --help' command to get more arguments infomation
python train.py \
--learning_rate=0.001 \
--use_cuda=true \
--num_passes=10 \
2>&1 | tee train.log
