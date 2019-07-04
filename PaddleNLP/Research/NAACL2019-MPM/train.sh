#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
output_dir=./output
prob_dir=./probs
bert_dir=./uncased_L-24_H-1024_A-16
mkdir -p $output_dir
mkdir -p $prob_dir

for model_type in raw cnn gru ffa
do
    python run_classifier.py \
            --bert_config_path ${bert_dir}/bert_config.json \
            --checkpoints ${output_dir}/bert_large_${model_type} \
            --init_pretraining_params ${bert_dir}/params \
            --data_dir ./data/Subtask-A \
            --vocab_path ${bert_dir}/vocab.txt \
            --task_name sem \
            --sub_model_type ${model_type} \
            --max_seq_len 128 \
            --batch_size 32 \
            --random_seed 777 \
            --save_steps 200 \
            --validation_steps 200 \
            --drop_keyword True 

    mv ${output_dir}/bert_large_${model_type}/prob.txt ${prob_dir}/prob_${model_type}.txt
done

