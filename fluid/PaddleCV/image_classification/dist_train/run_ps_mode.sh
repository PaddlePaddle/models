#!/bin/bash

export MODEL="DistResNet"
export PADDLE_PSERVER_ENDPOINTS="127.0.0.1:7160,127.0.0.1:7161"
export PADDLE_TRAINERS_NUM="2"

mkdir -p logs

PADDLE_TRAINING_ROLE="PSERVER" \
PADDLE_CURRENT_ENDPOINT="127.0.0.1:7160" \
python dist_train.py --model $MODEL --update_method pserver --batch_size 32 &> logs/ps0.log &

PADDLE_TRAINING_ROLE="PSERVER" \
PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
python dist_train.py --model $MODEL --update_method pserver --batch_size 32 &> logs/ps1.log &

PADDLE_TRAINING_ROLE="TRAINER" \
PADDLE_CURRENT_ENDPOINT="127.0.0.1:7160" \
PADDLE_TRAINER_ID="0" \
CUDA_VISIBLE_DEVICES="0" \
python dist_train.py --model $MODEL --update_method pserver --batch_size 32 &> logs/tr0.log &

PADDLE_TRAINING_ROLE="TRAINER" \
PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
PADDLE_TRAINER_ID="1" \
CUDA_VISIBLE_DEVICES="1" \
python dist_train.py --model $MODEL --update_method pserver --batch_size 32 &> logs/tr1.log &
