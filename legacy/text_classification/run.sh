#!/bin/sh

python train.py \
--nn_type="dnn" \
--batch_size=64 \
--num_passes=10 \
2>&1 | tee train.log
