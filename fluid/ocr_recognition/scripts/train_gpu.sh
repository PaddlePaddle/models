#!/bin/bash
OMP_NUM_THREADS=56 KMP_AFFINITY=granularity=fine,compact,1,0 LD_LIBRARY_PATH=/usr/local/lib python ../ctc_train.py \
    --use_gpu True \
    --parallel True \
    --batch_size 32 \
    --save_model_period 1 \
    --total_step 1 \
    --save_model_dir gpu_model