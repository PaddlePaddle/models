#!/usr/bin/env bash
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
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

set -xe

export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1

# set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0

# path of pre_train model
ERNIE_BASE_PATH=ernie_model
# path to save checkpoint
CHECKPOINT_PATH=output/
mkdir -p $CHECKPOINT_PATH
# path of train and dev data
DATA_PATH_train=data/train
DATA_PATH_dev=data/dev

# fine-tune params
python -u src/run_mrqa.py --use_cuda true\
        --batch_size 4 \
        --in_tokens false \
        --init_pretraining_params ${ERNIE_BASE_PATH}/params \
        --checkpoints ${CHECKPOINT_PATH} \
        --vocab_path ${ERNIE_BASE_PATH}/vocab.txt \
        --do_train true \
        --do_predict true \
        --save_steps 10000 \
        --warmup_proportion 0.1 \
        --weight_decay  0.01 \
        --epoch 2 \
        --max_seq_len 512 \
        --bert_config_path ${ERNIE_BASE_PATH}/ernie_config.json \
        --predict_file ${DATA_PATH_dev}/mrqa-combined.raw.json \
        --do_lower_case true \
        --doc_stride 128 \
        --train_file ${DATA_PATH_train}/mrqa-combined.raw.json \
        --learning_rate 3e-5 \
        --lr_scheduler linear_warmup_decay \
        --skip_steps 200 
