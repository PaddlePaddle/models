#!/bin/bash

echo "WARNING: This script only for run PaddlePaddle Fluid on one node..."
echo ""

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export PADDLE_PSERVER_PORTS=36001,36002
export PADDLE_PSERVER_PORT_ARRAY=(36001 36002)
export PADDLE_PSERVERS=2

export PADDLE_IP=127.0.0.1
export PADDLE_TRAINERS=2

export CPU_NUM=2
export NUM_THREADS=2
export PADDLE_SYNC_MODE=TRUE
export PADDLE_IS_LOCAL=0

export FLAGS_rpc_deadline=3000000
export GLOG_logtostderr=1


export TRAIN_DATA=data/enwik8
export DICT_PATH=data/enwik8_dict
export IS_SPARSE="--is_sparse"


echo "Start PSERVER ..."
for((i=0;i<$PADDLE_PSERVERS;i++))
do
    cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
    echo "PADDLE WILL START PSERVER "$cur_port
    GLOG_v=0 PADDLE_TRAINING_ROLE=PSERVER CUR_PORT=$cur_port PADDLE_TRAINER_ID=$i python -u train.py $IS_SPARSE &> pserver.$i.log &
done

echo "Start TRAINER ..."
for((i=0;i<$PADDLE_TRAINERS;i++))
do
    echo "PADDLE WILL START Trainer "$i
    GLOG_v=0 PADDLE_TRAINER_ID=$i PADDLE_TRAINING_ROLE=TRAINER python -u train.py $IS_SPARSE --train_data_path $TRAIN_DATA --dict_path $DICT_PATH &> trainer.$i.log &
done