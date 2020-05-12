#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

function run_train ()
{
    echo "training"
    python train.py \
        --data_path data/simple-examples/data/ \
        --model_type small \
        --use_gpu True \
        #--init_from_pretrain_model models/0/params
}

run_train
