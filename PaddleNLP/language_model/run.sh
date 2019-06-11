#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

function run_train() {
    echo "training"
    python train.py \
        --data_path data/simple-examples/data/ \
        --model_type small \
        --use_gpu True
}

run_train