#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
MODE=$2

# MODE be one of ['lite_train_lite_infer'，'lite_train_whole_infer' 
#                  'whole_train_whole_infer', 'whole_infer']          

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[12]}")


if [ ${MODE} = "lite_train_lite_infer" ];then
    # prepare lite data
    tar -xf ./test_images/lite_data.tar
    ln -s ./lite_data/ ./data
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./pretrain_models/ https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams  --no-check-certificate
    fi

elif [ ${MODE} = "whole_train_whole_infer" ];then
    tar -xf ../test_images/lite_data.tar
    # prepare whole data
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./pretrain_models/ https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams  --no-check-certificate
    fi
    
elif [ ${MODE} = "lite_train_whole_infer" ];then
    tar -xf ../test_images/lite_data.tar
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./pretrain_models/ https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams  --no-check-certificate
    fi

elif [ ${MODE} = "whole_infer" ];then
    tar -xf ../test_images/lite_data.tar
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./pretrain_models/ https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams  --no-check-certificate
    fi

elif [ ${MODE} = "serving_infer" ];then
    # get data
    tar -xf ./test_images/lite_data.tar
    # wget model
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./inference https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_infer.tar  --no-check-certificate
        cd ./inference && tar xf mobilenet_v3_small_infer.tar && cd ../
    fi
elif [ ${MODE} = "paddle2onnx_infer" ];then
    # get data
    tar -xf ./test_images/lite_data.tar
    # get model
    if [[ ${model_name} == "mobilenet_v3_small" ]];then
        wget -nc -P  ./inference https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_infer.tar  --no-check-certificate
        cd ./inference && tar xf mobilenet_v3_small_infer.tar && cd ../
    fi
fi
