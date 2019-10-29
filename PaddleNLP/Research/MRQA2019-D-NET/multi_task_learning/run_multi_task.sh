#!/bin/bash

# for gpu memory optimization
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u mtl_run.py

