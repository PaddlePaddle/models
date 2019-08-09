#!/bin/bash
set -xe

# Paddle debug envs
export GLOG_v=1
export GLOG_logtostderr=1
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.95

# Unset proxy
unset https_proxy http_proxy

# NCCL debug envs
export NCCL_DEBUG=INFO

nohup ./train.sh  > local.log 2>&1 &
