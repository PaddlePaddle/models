#!/bin/bash
set -xe

cluster_node_ips="127.0.0.1"
node_ip="127.0.0.1"

echo $#
distributed_args=""
if [[ $# -ge 1 ]]; then
    case "$1" in
    -cluster_node_ips) cluster_node_ips="$2" ; shift 2 ;;
    -node_ip) node_ip="$2" ; shift 2 ;;
    *)
       if [[ ${#1} > 0 ]]; then
          echo "not supported arugments ${1}" ; exit 1 ;
       else
           break
       fi
       ;;
  esac
  distributed_args="--cluster_node_ips ${cluster_node_ips} --node_ip ${node_ip}"
fi


# pretrain config
SAVE_STEPS=10000
BATCH_SIZE=4096
LR_RATE=1e-4
WEIGHT_DECAY=0.01
MAX_LEN=512
TRAIN_DATA_DIR=data/train
VALIDATION_DATA_DIR=data/validation
CONFIG_PATH=data/demo_config/bert_config.json
VOCAB_PATH=data/demo_config/vocab.txt

# Change your train arguments:
nohup python -m paddle.distributed.launch ${distributed_args}  --log_dir log \
        ./train.py \
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
        --loss_scaling 8.0 &
       
