#!/bin/bash
# ==============================================================================
# Copyright 2019 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8

if [ ! -d sqd_nell_first_stage_log ]; then
mkdir sqd_nell_first_stage_log
else
rm -r sqd_nell_first_stage_log/*
fi

if [ ! -d sqd_nell_first_stage_output ]; then
mkdir sqd_nell_first_stage_output
else
rm -r sqd_nell_first_stage_output/*
fi

export FLAGS_cudnn_deterministic=true
export FLAGS_cpu_deterministic=true

PWD_DIR=`pwd`
DATA=../data/
BERT_DIR=cased_L-24_H-1024_A-16
CPT_EMBEDDING_PATH=../retrieve_concepts/KB_embeddings/nell_concept2vec.txt

python3 src/run_squad.py \
  --batch_size 6 \
  --do_train true \
  --do_predict true \
  --use_ema false \
  --do_lower_case false \
  --init_pretraining_params $BERT_DIR/params \
  --train_file $DATA/SQuAD/train-v1.1.json \
  --predict_file $DATA/SQuAD/dev-v1.1.json \
  --vocab_path $BERT_DIR/vocab.txt \
  --bert_config_path $BERT_DIR/bert_config.json \
  --freeze true \
  --save_steps 4000 \
  --weight_decay 0.01 \
  --warmup_proportion 0.0 \
  --learning_rate 3e-5 \
  --epoch 1 \
  --max_seq_len 384 \
  --doc_stride 128 \
  --concept_embedding_path $CPT_EMBEDDING_PATH \
  --use_nell true \
  --random_seed 45 \
  --checkpoints sqd_nell_first_stage_output/ 1>$PWD_DIR/sqd_nell_first_stage_log/train.log 2>&1
