#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

#MobileNet v1:
python quant.py \
       --model=MobileNet \
       --pretrained_fp32_model=../data/pretrain/MobileNetV1_pretrained \
       --use_gpu=True \
       --data_dir=../data/ILSVRC2012 \
       --batch_size=256 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --num_epochs=20 \
       --lr=0.0001 \
       --act_quant_type=abs_max \
       --wt_quant_type=abs_max


#ResNet50:
#python quant.py \
#       --model=ResNet50 \
#       --pretrained_fp32_model=../data/pretrain/ResNet50_pretrained \
#       --use_gpu=True \
#       --data_dir=../data/ILSVRC2012 \
#       --batch_size=128 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --lr_strategy=piecewise_decay \
#       --num_epochs=20 \
#       --lr=0.0001 \
#       --act_quant_type=abs_max \
#       --wt_quant_type=abs_max

