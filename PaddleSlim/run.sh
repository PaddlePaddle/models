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

# enable GC strategy
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0

# for distillation
#-----------------
export CUDA_VISIBLE_DEVICES=0,1,2,3


# Fixing name conflicts in distillation
mv ResNet50_pretrained/conv1_weights ResNet50_pretrained/res_conv1_weights
mv ResNet50_pretrained/fc_0.w_0 ResNet50_pretrained/res_fc.w_0
mv ResNet50_pretrained/fc_0.b_0 ResNet50_pretrained/res_fc.b_0
python compress.py \
--model "MobileNet" \
--teacher_model "ResNet50" \
--teacher_pretrained_model ./pretrain/ResNet50_pretrained \
--compress_config ./configs/mobilenetv1_resnet50_distillation.yaml

mv ResNet50_pretrained/res_conv1_weights ResNet50_pretrained/conv1_weights
mv ResNet50_pretrained/res_fc.w_0 ResNet50_pretrained/fc_0.w_0
mv ResNet50_pretrained/res_fc.b_0 ResNet50_pretrained/fc_0.b_0

# for sensitivity filter pruning
#-------------------------------
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
#-----------------
#export CUDA_VISIBLE_DEVICES=0
#python compress.py \
#--batch_size 64 \
#--model "MobileNet" \
#--pretrained_model ./pretrain/MobileNetV1_pretrained \
#--compress_config ./configs/quantization.yaml

# for distillation with quantization
#-----------------------------------
#export CUDA_VISIBLE_DEVICES=0
#
## Fixing name conflicts in distillation
#mv ResNet50_pretrained/conv1_weights ResNet50_pretrained/res_conv1_weights
#mv ResNet50_pretrained/fc_0.w_0 ResNet50_pretrained/res_fc.w_0
#mv ResNet50_pretrained/fc_0.b_0 ResNet50_pretrained/res_fc.b_0
#
#python compress.py \
#--model "MobileNet" \
#--teacher_model "ResNet50" \
#--teacher_pretrained_model ./data/pretrain/ResNet50_pretrained \
#--compress_config ./configs/quantization_dist.yaml
#
#mv ResNet50_pretrained/res_conv1_weights ResNet50_pretrained/conv1_weights
#mv ResNet50_pretrained/res_fc.w_0 ResNet50_pretrained/fc_0.w_0
#mv ResNet50_pretrained/res_fc.b_0 ResNet50_pretrained/fc_0.b_0

# for uniform filter pruning with quantization
#---------------------------------------------
#export CUDA_VISIBLE_DEVICES=0
#python compress.py \
#--model "MobileNet" \
#--pretrained_model ./data/pretrain/MobileNetV1_pretrained \
#--compress_config ./configs/quantization_pruning.yaml

