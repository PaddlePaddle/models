#!/bin/bash
set -x 
export CUDA_VISIBLE_DEVICES=6

python train.py \
        --src_lang en --tar_lang vi \
        --attention False \
        --num_layers 2 \
        --hidden_size 512 \
        --src_vocab_size 17191 \
        --tar_vocab_size 7709 \
        --batch_size 128 \
        --dropout 0.2 \
        --init_scale  0.1 \
        --max_grad_norm 5.0 \
        --train_data_prefix data/en-vi/train \
        --eval_data_prefix data/en-vi/tst2012 \
        --test_data_prefix data/en-vi/tst2013 \
        --vocab_prefix data/en-vi/vocab \
        --use_gpu True \
        --model_path base_models

python infer.py \
    --attention False \
    --src_lang en --tar_lang vi \
    --num_layers 2 \
    --hidden_size 512 \
    --src_vocab_size 17191 \
    --tar_vocab_size 7709 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --vocab_prefix data/en-vi/vocab \
    --infer_file data/en-vi/tst2013.en \
    --reload_model base_models/epoch_10 \
    --infer_output_file attention_infer_output/infer_output.txt \
    --beam_size 10 \
    --use_gpu True

python train.py \
        --src_lang en --tar_lang vi \
        --attention True \
        --num_layers 2 \
        --hidden_size 512 \
        --src_vocab_size 17191 \
        --tar_vocab_size 7709 \
        --batch_size 128 \
        --dropout 0.2 \
        --init_scale  0.1 \
        --max_grad_norm 5.0 \
        --train_data_prefix data/en-vi/train \
        --eval_data_prefix data/en-vi/tst2012 \
        --test_data_prefix data/en-vi/tst2013 \
        --vocab_prefix data/en-vi/vocab \
        --use_gpu True \
        --model_path attention_models

python infer.py \
    --attention True \
    --src_lang en --tar_lang vi \
    --num_layers 2 \
    --hidden_size 512 \
    --src_vocab_size 17191 \
    --tar_vocab_size 7709 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --vocab_prefix data/en-vi/vocab \
    --infer_file data/en-vi/tst2013.en \
    --reload_model attention_models/epoch_10 \
    --infer_output_file attention_infer_output/infer_output.txt \
    --beam_size 10 \
    --use_gpu True