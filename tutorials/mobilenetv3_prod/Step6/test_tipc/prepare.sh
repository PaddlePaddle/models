#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# just support 'lite_train_lite_infer' now

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

if [ ${MODE} = "lite_train_lite_infer" ];then
    # pretrain lite train data
    tar -xf test_tipc/data/lite_data.tar
fi



