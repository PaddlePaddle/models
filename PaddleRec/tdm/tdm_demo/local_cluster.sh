#!/bin/bash
echo "WARNING: This script only for run PaddlePaddle Fluid on one node"

CLUSTER_DIRS="./"

if [ ! -d "${CLUSTER_DIRS}/model" ]; then
  mkdir "${CLUSTER_DIRS}/model"
  echo "Create model folder for store infer model"
fi

if [ ! -d "${CLUSTER_DIRS}/log" ]; then
  mkdir "${CLUSTER_DIRS}/log"
  echo "Create log floder for store running log"
fi

if [ ! -d "${CLUSTER_DIRS}/output" ]; then
  mkdir "${CLUSTER_DIRS}/output"
  echo "Create output floder"
fi

# environment variables for fleet distribute training
export PADDLE_TRAINER_ID=0

export PADDLE_TRAINERS_NUM=1
export OUTPUT_PATH="output"
export SYS_JOB_ID="test"

export FLAGS_communicator_thread_pool_size=5
export FLAGS_communicator_fake_rpc=0
export FLAGS_communicator_is_sgd_optimizer=0

export FLAGS_communicator_send_queue_size=1
export FLAGS_communicator_max_merge_var_num=1
export FLAGS_communicator_max_send_grad_num_before_recv=1
export FLAGS_communicator_min_send_grad_num_before_recv=1
export FLAGS_rpc_retry_times=3
export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:36001"
export PADDLE_PSERVER_PORT_ARRAY=(36001)
export PADDLE_PSERVER_NUMS=1
export PADDLE_TRAINERS=1

export TRAINING_ROLE=PSERVER
export GLOG_v=0
export GLOG_logtostderr=0
ps -ef|grep python|awk '{print $2}'|xargs kill -9

train_mode=$1

for((i=0;i<$PADDLE_PSERVER_NUMS;i++))
do
    cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
    echo "PADDLE WILL START PSERVER "$cur_port
    export PADDLE_PORT=${cur_port}
    export POD_IP=127.0.0.1
    sh ./async_train.sh &> ./log/pserver.$i.log &
done

export TRAINING_ROLE=TRAINER
export GLOG_v=0
export GLOG_logtostderr=0

for((i=0;i<$PADDLE_TRAINERS;i++))
do
    echo "PADDLE WILL START Trainer "$i
    PADDLE_TRAINER_ID=$i
    sh ./async_train.sh &> ./log/trainer.$i.log &
done

echo "Training log stored in ./log/"
