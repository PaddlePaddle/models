#!/usr/bin/env bash

# download pretrain model
root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
MobileNetV1="MobileNetV1_pretrained.zip"
ResNet50="ResNet50_pretrained.zip"
pretrain_dir='./pretrain'

if [ ! -d ${pretrain_dir} ]; then
  mkdir ${pretrain_dir}
fi

cd ${pretrain_dir}

if [ ! -f ${MobileNetV1} ]; then
    wget ${root_url}/${MobileNetV1}
    unzip ${MobileNetV1}
fi

if [ ! -f ${ResNet50} ]; then
    wget ${root_url}/${ResNet50}
    unzip ${ResNet50}
fi

cd -

# for distillation
#--------------------
export CUDA_VISIBLE_DEVICES=0
python compress.py \
--model "MobileNet" \
--teacher_model "ResNet50" \
--teacher_pretrained_model ./pretrain/ResNet50_pretrained \
--compress_config ./configs/mobilenetv1_resnet50_distillation.yaml


# for sensitivity filter pruning
#---------------------------
#export CUDA_VISIBLE_DEVICES=0
#python compress.py \
#--model "MobileNet" \
#--pretrained_model ./pretrain/MobileNetV1_pretrained \
#--compress_config ./configs/filter_pruning_sen.yaml

# for uniform filter pruning
#---------------------------
#export CUDA_VISIBLE_DEVICES=0
#python compress.py \
#--model "MobileNet" \
#--pretrained_model ./pretrain/MobileNetV1_pretrained \
#--compress_config ./configs/filter_pruning_uniform.yaml

# for quantization
#---------------------------
#export CUDA_VISIBLE_DEVICES=0
#python compress.py \
#--batch_size 64 \
#--model "MobileNet" \
#--pretrained_model ./pretrain/MobileNetV1_pretrained \
#--compress_config ./configs/quantization.yaml

