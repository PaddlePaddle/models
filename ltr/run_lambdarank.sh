#!/bin/sh

python lambda_rank.py \
       --run_type="train" \
       --num_passes=10 \
       2>&1 | tee lambdarank_train.log

python lambda_rank.py \
       --run_type="infer" \
       --num_passes=10 \
       2>&1 | tee lambdarank_infer.log
