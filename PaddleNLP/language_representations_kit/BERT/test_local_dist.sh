#!/bin/bash
set -xe

# Paddle debug envs
export GLOG_v=1
#export GLOG_vmodule=fused_all_reduce_op_handle=10,all_reduce_op_handle=10,alloc_continuous_space_op=10,fuse_all_reduce_op_pass=10,alloc_continuous_space_for_grad_pass=10
export GLOG_logtostderr=1
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.95

# Unset proxy
unset https_proxy http_proxy

# NCCL debug envs
export NCCL_DEBUG=INFO

# pretrain config
SAVE_STEPS=10000
BATCH_SIZE=1024
LR_RATE=1e-4
WEIGHT_DECAY=0.01
MAX_LEN=512
TRAIN_DATA_DIR=data/train
VALIDATION_DATA_DIR=data/validation
CONFIG_PATH=data/demo_config/bert_config.json
VOCAB_PATH=data/demo_config/vocab.txt

# Change your train arguments:
nohup python -m paddle.distributed.launch ${distributed_args} --log_dir log \
        ./train.py ${is_distributed}\
        --use_cuda true\
        --weight_sharing true\
        --batch_size ${BATCH_SIZE} \
        --data_dir ${TRAIN_DATA_DIR} \
        --validation_set_dir ${VALIDATION_DATA_DIR} \
        --bert_config_path ${CONFIG_PATH} \
        --vocab_path ${VOCAB_PATH} \
        --generate_neg_sample true\
        --checkpoints ./output \
        --save_steps ${SAVE_STEPS} \
        --learning_rate ${LR_RATE} \
        --weight_decay ${WEIGHT_DECAY:-0} \
        --max_seq_len ${MAX_LEN} \
        --skip_steps 20 \
        --validation_steps 1000 \
        --num_iteration_per_drop_scope 10 \
        --use_fp16 false \
        --loss_scaling 8.0 2>&1 &
       


# Comment it if your nccl support IB
#export NCCL_IB_DISABLE=1
#
## Add your nodes endpoints here.
#export worker_endpoints=127.0.0.1:9184,127.0.0.1:9185
#export current_endpoint=127.0.0.1:9184
#export CUDA_VISIBLE_DEVICES=0
#
#./train.sh -local n > 0.log 2>&1 &
#
## Add your nodes endpoints here.
#export current_endpoint=127.0.0.1:9185
#export CUDA_VISIBLE_DEVICES=1
#
#./train.sh -local n > 1.log 2>&1 &
