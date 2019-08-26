#!/bin/bash

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8

if [ ! -d log ]; then
mkdir log
else
rm -r log/*
fi

if [ ! -d output ]; then
mkdir output
else
rm -r output/*
fi

export FLAGS_cudnn_deterministic=true
export FLAGS_cpu_deterministic=true

DATA=../data/
BERT_DIR=cased_L-24_H-1024_A-16
WN_CPT_EMBEDDING_PATH=../retrieve_concepts/KB_embeddings/wn_concept2vec.txt
NELL_CPT_EMBEDDING_PATH=../retrieve_concepts/KB_embeddings/nell_concept2vec.txt

python3 src/run_record_twomemory.py \
  --batch_size 6 \
  --do_train true \
  --do_predict true \
  --do_lower_case false \
  --init_pretraining_params $BERT_DIR/params \
  --train_file $DATA/ReCoRD/train.json \
  --predict_file $DATA/ReCoRD/dev.json \
  --vocab_path $BERT_DIR/vocab.txt \
  --bert_config_path $BERT_DIR/bert_config.json \
  --freeze false \
  --save_steps 4000 \
  --validation_steps 4000 \
  --weight_decay 0.01 \
  --warmup_proportion 0.1 \
  --learning_rate 3e-5 \
  --epoch 4 \
  --max_seq_len 384 \
  --doc_stride 128 \
  --wn_concept_embedding_path $WN_CPT_EMBEDDING_PATH \
  --nell_concept_embedding_path $NELL_CPT_EMBEDDING_PATH \
  --use_wordnet true \
  --use_nell true \
  --random_seed 45 \
  --checkpoints output/ 1>$PWD_DIR/log/train.log 2>&1