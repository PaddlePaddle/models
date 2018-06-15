#!/bin/bash

#
#function:
#   a tool used to compare the results produced by paddle and caffe
#

if [[ $# -lt 2 ]];then
    echo "usage:"
    echo "  bash $0 [model_name] [param_name] [caffe_name]"
    exit 1
fi

model_name=$1
param_name=$2
paddle_file="./results/${model_name}.paddle/${param_name}.npy"
if [[ $# -eq 3 ]];then
    caffe_file="./results/${model_name}.caffe/${3}.npy"
else
    caffe_file="./results/${model_name}.caffe/${2}.npy"
fi
cmd="python ./compare.py $paddle_file $caffe_file"
echo $cmd
eval $cmd
