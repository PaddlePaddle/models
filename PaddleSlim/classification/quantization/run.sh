#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=0

# for quantization for mobilenet_v1
#python compress.py \
#    --model "MobileNet" \
#    --use_gpu 1 \
#    --batch_size 32 \
#    --pretrained_model ../pretrain/MobileNetV1_pretrained \
#    --config_file "./configs/mobilenet_v1.yaml" \
#> mobilenet_v1.log 2>&1 &
#tailf mobilenet_v1.log

# for quantization of mobilenet_v2
# python compress.py \
#    --model "MobileNetV2" \
#    --use_gpu 1 \
#    --batch_size 32 \
#    --pretrained_model ../pretrain/MobileNetV2_pretrained \
#    --config_file "./configs/mobilenet_v2.yaml" \
#    > mobilenet_v2.log 2>&1 &
#tailf mobilenet_v2.log


# for compression of resnet50
python compress.py \
    --model "ResNet50" \
    --use_gpu 1 \
    --batch_size 32 \
    --pretrained_model ../pretrain/ResNet50_pretrained \
    --config_file "./configs/resnet50.yaml" \
    > resnet50.log 2>&1 &
tailf resnet50.log

