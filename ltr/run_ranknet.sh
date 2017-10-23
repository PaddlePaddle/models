#!/bin/sh

python ranknet.py \
       --run_type="train" \
       --num_passes=10 \
       2>&1 | tee ranknet_train.log

python ranknet.py \
       --run_type="infer" \
       --num_passes=10 \
       2>&1 | tee ranknet_infer.log
