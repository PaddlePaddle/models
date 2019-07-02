#!/bin/bash

train_atis_slot(){
  python -u train.py \
  --task_name atis_slot \
  --use_cuda true \
  --do_train true \
  --do_val true \
  --do_test true \
  --epoch 2 \
  --batch_size 32 \
  --data_dir ./data/atis/atis_slot \
  --bert_config_path ./uncased_L-12_H-768_A-12/bert_config.json \
  --vocab_path ./uncased_L-12_H-768_A-12/vocab.txt \
  --init_pretraining_params ./uncased_L-12_H-768_A-12/params \
  --checkpoints ./output/atis_slot \
  --save_steps 100 \
  --learning_rate 2e-5 \
  --weight_decay  0.01 \
  --max_seq_len 128 \
  --skip_steps 10 \
  --validation_steps 1000000 \
  --num_iteration_per_drop_scope 10 \
  --use_fp16 false \
  --enable_ce
}

train_mrda(){
  python -u train.py \
  --task_name mrda \
  --use_cuda true \
  --do_train true \
  --do_val true \
  --do_test true \
  --epoch 2 \
  --batch_size 4096 \
  --data_dir ./data/mrda \
  --bert_config_path ./uncased_L-12_H-768_A-12/bert_config.json \
  --vocab_path ./uncased_L-12_H-768_A-12/vocab.txt \
  --init_pretraining_params ./uncased_L-12_H-768_A-12/params \
  --checkpoints ./output/mrda \
  --save_steps 500 \
  --learning_rate 2e-5 \
  --weight_decay  0.01 \
  --max_seq_len 128 \
  --skip_steps 200 \
  --validation_steps 1000000 \
  --num_iteration_per_drop_scope 10 \
  --use_fp16 false \
  --enable_ce
}


cudaid=${multi:=0,1,2,3}
export CUDA_VISIBLE_DEVICES=$cudaid
train_atis_slot | python _ce.py
sleep 20

cudaid=${single:=0}
export CUDA_VISIBLE_DEVICES=$cudaid
train_atis_slot | python _ce.py

