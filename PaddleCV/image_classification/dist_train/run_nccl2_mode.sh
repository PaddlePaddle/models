#!/bin/bash

export MODEL="DistResNet"
export PADDLE_TRAINER_ENDPOINTS="127.0.0.1:7160,127.0.0.1:7161"
# PADDLE_TRAINERS_NUM is used only for reader when nccl2 mode
export PADDLE_TRAINERS_NUM="2"

mkdir -p logs

# NOTE: set NCCL_P2P_DISABLE so that can run nccl2 distribute train on one node.

PADDLE_TRAINING_ROLE="TRAINER" \
PADDLE_CURRENT_ENDPOINT="127.0.0.1:7160" \
PADDLE_TRAINER_ID="0" \
CUDA_VISIBLE_DEVICES="0" \
NCCL_P2P_DISABLE="1" \
python dist_train.py --model $MODEL --update_method nccl2 --batch_size 32 &> logs/tr0.log &

PADDLE_TRAINING_ROLE="TRAINER" \
PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
PADDLE_TRAINER_ID="1" \
CUDA_VISIBLE_DEVICES="1" \
NCCL_P2P_DISABLE="1" \
python dist_train.py --model $MODEL --update_method nccl2 --batch_size 32 &> logs/tr1.log &
