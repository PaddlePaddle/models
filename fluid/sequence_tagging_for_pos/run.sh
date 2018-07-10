#!/usr/bin/env sh

#you can use 'python train.py --help' command to get more arguments infomation
python train.py \
--nn_type="window" \
--window_size=5 \
--learning_rate=0.001 \
--batch_size=128 \
--use_cuda=false \
--num_passes=10 \
2>&1 | tee train.log
