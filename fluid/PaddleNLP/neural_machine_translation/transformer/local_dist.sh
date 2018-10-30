#!/bin/bash

set -x

unset http_proxy
unset https_proxy

#pserver
export TRAINING_ROLE=PSERVER
export PADDLE_PORT=30134
export PADDLE_PSERVERS=127.0.0.1
export PADDLE_IS_LOCAL=0
export PADDLE_INIT_TRAINER_COUNT=1
export POD_IP=127.0.0.1
export PADDLE_TRAINER_ID=0
export PADDLE_TRAINERS_NUM=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/:/workspace/brpc
export PYTHONPATH=$PYTHONPATH:/paddle/build/build_reader_RelWithDebInfo_gpu/python

#GLOG_v=7 GLOG_logtostderr=1 
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u train.py \
    --src_vocab_fpath 'cluster_test_data_en_fr/thirdparty/vocab.wordpiece.en-fr' \
    --trg_vocab_fpath 'cluster_test_data_en_fr/thirdparty/vocab.wordpiece.en-fr' \
    --special_token '<s>' '<e>' '<unk>'  \
    --token_delimiter '\x01' \
    --train_file_pattern 'cluster_test_data_en_fr/train/train.wordpiece.en-fr.0' \
    --val_file_pattern 'cluster_test_data_en_fr/thirdparty/newstest2014.wordpiece.en-fr' \
    --use_token_batch True \
    --batch_size  3200 \
    --sort_type pool \
    --pool_size 200000 \
    --local False > pserver.log 2>&1 &

pserver_pid=$(echo $!)
echo $pserver_pid

sleep 30s

#trainer
export TRAINING_ROLE=TRAINER
export PADDLE_PORT=30134
export PADDLE_PSERVERS=127.0.0.1
export PADDLE_IS_LOCAL=0
export PADDLE_INIT_TRAINER_COUNT=1
export POD_IP=127.0.0.1
export PADDLE_TRAINER_ID=0
export PADDLE_TRAINERS_NUM=1

CUDA_VISIBLE_DEVICES=4,5,6,7 python -u train.py \
    --src_vocab_fpath 'cluster_test_data_en_fr/thirdparty/vocab.wordpiece.en-fr' \
    --trg_vocab_fpath 'cluster_test_data_en_fr/thirdparty/vocab.wordpiece.en-fr' \
    --special_token '<s>' '<e>' '<unk>'  \
    --token_delimiter '\x01' \
    --train_file_pattern 'cluster_test_data_en_fr/train/train.wordpiece.en-fr.0' \
    --val_file_pattern 'cluster_test_data_en_fr/thirdparty/newstest2014.wordpiece.en-fr' \
    --use_token_batch True \
    --batch_size  3200 \
    --sort_type pool \
    --pool_size 200000 \
    --local False > trainer.log 2>&1 &

#sleep 80
#kill -9 $pserver_pid
