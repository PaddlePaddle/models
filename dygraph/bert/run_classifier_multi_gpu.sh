#!/bin/bash

BERT_BASE_PATH="./data/pretrained_models/uncased_L-12_H-768_A-12/"
TASK_NAME='MNLI'
DATA_PATH="./data/glue_data/MNLI/"
CKPT_PATH="./data/saved_model/mnli_models"
GPU_TO_USE=0,1,2,3

# start fine-tuning
python -m paddle.distributed.launch --selected_gpus=$GPU_TO_USE --log_dir ./cls_log run_classifier.py \
    --task_name ${TASK_NAME} \
    --use_cuda true \
    --use_data_parallel true \
    --do_train true \
    --do_test true \
    --batch_size 64 \
    --in_tokens false \
    --init_pretraining_params ${BERT_BASE_PATH}/dygraph_params/ \
    --data_dir ${DATA_PATH} \
    --vocab_path ${BERT_BASE_PATH}/vocab.txt \
    --checkpoints ${CKPT_PATH} \
    --save_steps 1000 \
    --weight_decay  0.01 \
    --warmup_proportion 0.1 \
    --validation_steps 100 \
    --epoch 3 \
    --max_seq_len 128 \
    --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
    --learning_rate 5e-5 \
    --skip_steps 10 \
    --shuffle true


