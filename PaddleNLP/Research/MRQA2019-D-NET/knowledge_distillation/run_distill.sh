#!/bin/bash

export FLAGS_sync_nccl_allreduce=0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
if  [ ! "$CUDA_VISIBLE_DEVICES" ]
then
    export CPU_NUM=1
    use_cuda=false
else
    use_cuda=true
fi

# path of pre_train model
INPUT_PATH="data/input"
PRETRAIN_MODEL_PATH="data/pretrain_model/squad2_model"
# path to save checkpoint
CHECKPOINT_PATH="data/output/output_mrqa"
mkdir -p $CHECKPOINT_PATH

python -u train.py --use_cuda ${use_cuda}\
        --batch_size 8 \
        --in_tokens false \
        --init_pretraining_params ${PRETRAIN_MODEL_PATH}/params \
        --checkpoints $CHECKPOINT_PATH \
        --vocab_path ${PRETRAIN_MODEL_PATH}/vocab.txt \
        --do_distill true \
        --do_train true \
        --do_predict true \
        --save_steps 10000 \
        --warmup_proportion 0.1 \
        --weight_decay  0.01 \
        --sample_rate 0.02 \
        --epoch 2 \
        --max_seq_len 512 \
        --bert_config_path ${PRETRAIN_MODEL_PATH}/bert_config.json \
        --predict_file ${INPUT_PATH}/mrqa_distill_data/mrqa-combined.all_dev.raw.json \
        --do_lower_case false \
        --doc_stride 128 \
        --train_file ${INPUT_PATH}/mrqa_distill_data/mrqa_distill.json \
        --mlm_path ${INPUT_PATH}/mlm_data \
        --mix_ratio 2.0 \
        --learning_rate 3e-5 \
        --lr_scheduler linear_warmup_decay \
        --skip_steps 100 

