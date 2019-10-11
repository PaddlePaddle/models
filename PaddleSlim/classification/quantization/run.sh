#!/usr/bin/env bash

# download pretrain model
root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
MobileNetV1="MobileNetV1_pretrained.tar"
MobileNetV2="MobileNetV2_pretrained.tar"
ResNet34="ResNet34_pretrained.tar"
pretrain_dir='../pretrain'

if [ ! -d ${pretrain_dir} ]; then
  mkdir ${pretrain_dir}
fi

cd ${pretrain_dir}

if [ ! -f ${MobileNetV1} ]; then
    wget ${root_url}/${MobileNetV1}
    tar xf ${MobileNetV1}
fi

if [ ! -f ${MobileNetV2} ]; then
    wget ${root_url}/${MobileNetV2}
    tar xf ${MobileNetV2}
fi

if [ ! -f ${ResNet34} ]; then
    wget ${root_url}/${ResNet34}
    tar xf ${ResNet34}
fi

cd -

# enable GC strategy
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0

export CUDA_VISIBLE_DEVICES=0

## for quantization for mobilenet_v1
#python -u compress.py \
#    --model "MobileNet" \
#    --use_gpu 1 \
#    --batch_size 256 \
#    --pretrained_model ../pretrain/MobileNetV1_pretrained \
#    --config_file "./configs/mobilenet_v1.yaml" \ 
#> mobilenet_v1.log 2>&1 &
#tailf mobilenet_v1.log

## for quantization of mobilenet_v2
#python -u compress.py \
#    --model "MobileNetV2" \
#    --use_gpu 1 \
#    --batch_size 32 \
#    --pretrained_model ../pretrain/MobileNetV2_pretrained \
#    --config_file "./configs/mobilenet_v2.yaml" \
#    > mobilenet_v2.log 2>&1 &
#tailf mobilenet_v2.log

# for compression of resnet34
python -u compress.py \
    --model "ResNet34" \
    --use_gpu 1 \
    --batch_size 32 \
    --pretrained_model ../pretrain/ResNet34_pretrained \
    --config_file "./configs/resnet34.yaml" \
  > resnet34.log 2>&1 &
tailf resnet34.log
