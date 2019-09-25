#!/bin/bash

set -ex

model=ResNet50
batch_size=32


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

gpu_num=`echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}'`

total_batch_size=$(($batch_size*$gpu_num))

#ResNet50:
python train.py \
       --model=ResNet50 \
       --batch_size=${total_batch_size} \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --num_epochs=120 \
       --lr=0.1 \
       --l2_decay=1e-4 \
       --use_iterable_py_reader $@
