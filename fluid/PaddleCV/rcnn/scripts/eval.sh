#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model=$1 # faster_rcnn, mask_rcnn
if [ "$model" = "faster_rcnn" ]; then
  mask_on="--MASK_ON False"
elif [ "$model" = "mask_rcnn" ]; then
  mask_on="--MASK_ON True"
else
  echo "Invalid model provided. Please use one of {faster_rcnn, mask_rcnn}"
  exit 1
fi

python -u ../eval_coco_map.py \
    $mask_on \
    --pretrained_model=../output/model_iter179999 \
    --data_dir=../dataset/coco/ \
