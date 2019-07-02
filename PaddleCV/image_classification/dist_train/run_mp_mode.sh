#!/bin/bash

# Test using 4 GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MODEL="DistResNet"
export PADDLE_TRAINER_ENDPOINTS="127.0.0.1:7160,127.0.0.1:7161,127.0.0.1:7162,127.0.0.1:7163"
# PADDLE_TRAINERS_NUM is used only for reader when nccl2 mode
export PADDLE_TRAINERS_NUM="4"

mkdir -p logs

for i in {0..3}
do
PADDLE_TRAINING_ROLE="TRAINER" \
PADDLE_CURRENT_ENDPOINT="127.0.0.1:716${i}" \
PADDLE_TRAINER_ID="${i}" \
FLAGS_selected_gpus="${i}" \
python -u dist_train.py --model $MODEL --update_method nccl2 \
--batch_size 32 \
--fp16 0 --scale_loss 1 &> logs/tr$i.log &
done
