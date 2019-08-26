#!/bin/bash

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8

if [ ! -d record_wn_first_stage_log ]; then
mkdir record_wn_first_stage_log
else
rm -r record_wn_first_stage_log/*
fi

if [ ! -d record_wn_first_stage_output ]; then
mkdir record_wn_first_stage_output
else
rm -r record_wn_first_stage_output/*
fi

export FLAGS_cudnn_deterministic=true
export FLAGS_cpu_deterministic=true

PWD_DIR=`pwd`
DATA=../data/
BERT_DIR=cased_L-24_H-1024_A-16
CPT_EMBEDDING_PATH=../retrieve_concepts/KB_embeddings/wn_concept2vec.txt

python3 src/run_record.py \
  --batch_size 6 \
  --do_train true \
  --do_predict true \
  --use_ema false \
  --do_lower_case false \
  --init_pretraining_params $BERT_DIR/params \
  --train_file $DATA/ReCoRD/train.json \
  --predict_file $DATA/ReCoRD/dev.json \
  --vocab_path $BERT_DIR/vocab.txt \
  --bert_config_path $BERT_DIR/bert_config.json \
  --freeze true \
  --save_steps 4000 \
  --weight_decay 0.01 \
  --warmup_proportion 0.0 \
  --learning_rate 3e-4 \
  --epoch 10 \
  --max_seq_len 384 \
  --doc_stride 128 \
  --concept_embedding_path $CPT_EMBEDDING_PATH \
  --use_wordnet true \
  --random_seed 45 \
  --checkpoints record_wn_first_stage_output/ 1>$PWD_DIR/record_wn_first_stage_log/train.log 2>&1