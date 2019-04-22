#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model=$1 # faster_rcnn, mask_rcnn
if [ "$model" = "faster_rcnn" ]; then
  mask_on="--MASK_ON False"
elif [ "$model" = "mask_rcnn" ]; then
  mask_on="--MASK_ON True"
else
  echo "Invalid model provided. Please use one of {faster_rcnn, mask_rcnn}"
  exit 1
fi

python -u ../train.py \
    $mask_on \
    --model_save_dir=../output/ \
    --pretrained_model=../imagenet_resnet50_fusebn/ \
    --data_dir=../dataset/coco/ \  

