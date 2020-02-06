#!/bin/bash
export BASE_PATH="$PWD"

#NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
#export NCCL_IB_RETRY_CNT=0

#PADDLE
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=0.0

#Cudnn
#export FLAGS_cudnn_exhaustive_search=1
export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${BASE_PATH}/nccl_2.3.5/lib/:$LD_LIBRARY_PATH"
#proxy
unset https_proxy http_proxy

# GLOG
export GLOG_v=1
#export GLOG_vmodule=fused_all_reduce_op_handle=10,all_reduce_op_handle=10,alloc_continuous_space_op=10,fuse_all_reduce_op_pass=10,alloc_continuous_space_for_grad_pass=10,fast_threaded_ssa_graph_executor=10,threaded_ssa_graph_executor=10,backward_op_deps_pass=10,graph=10
export GLOG_logtostderr=1

