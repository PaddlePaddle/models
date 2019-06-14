#!/bin/bash

train(){

    DATA_PATH=./dataset/wmt16

    python  train.py \
          --src_vocab_fpath $DATA_PATH/en_10000.dict \
          --trg_vocab_fpath $DATA_PATH/de_10000.dict \
          --special_token '<s>' '<e>' '<unk>' \
          --train_file_pattern $DATA_PATH/wmt16/train \
          --val_file_pattern $DATA_PATH/wmt16/val \
          --use_token_batch True \
          --batch_size 1024 \
          --sort_type pool \
          --pool_size 200000 \
          --shuffle False \
          --enable_ce True \
          --local False \
          --shuffle_batch False \
          --use_py_reader True \
          --use_mem_opt True \
          --fetch_steps 100  $@ \
          dropout_seed 10 \
          learning_rate 2.0 \
          warmup_steps 8000 \
          beta2 0.997 \
          d_model 512 \
          d_inner_hid 2048 \
          n_head 8 \
          prepostprocess_dropout 0.1 \
          attention_dropout 0.1 \
          relu_dropout 0.1 \
          weight_sharing True \
          pass_num 2 \
          model_dir 'tmp_models' \
          ckpt_dir 'tmp_ckpts' &
}

export PADDLE_PSERVERS="127.0.0.1:7160,127.0.0.1:7161"
export PADDLE_TRAINERS_NUM="2"
mkdir -p logs

run_ps_ce_card1(){
    TRAINING_ROLE="PSERVER" \
    PADDLE_CURRENT_ENDPOINT="127.0.0.1:7160" \
    FLAGS_fraction_of_gpu_memory_to_use=0.0 \
    train &> logs/ps0.log &

    TRAINING_ROLE="PSERVER" \
    PADDLE_CURRENT_ENDPOINT="127.0.0.1:7161" \
    FLAGS_fraction_of_gpu_memory_to_use=0.0 \
    train &> logs/ps1.log &

    TRAINING_ROLE="TRAINER" \
    PADDLE_CURRENT_ENDPOINT="127.0.0.1:7162" \
    PADDLE_TRAINER_ID="0" \
    CUDA_VISIBLE_DEVICES="6" \
    train &> logs/tr0.log|python _ce.py &

    TRAINING_ROLE="TRAINER" \
    PADDLE_CURRENT_ENDPOINT="127.0.0.1:7163" \
    PADDLE_TRAINER_ID="1" \
    CUDA_VISIBLE_DEVICES="7" \
    train &> logs/tr1.log |python _ce.py &
}

run_ps_ce_card1
