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

python -u ../infer.py \
    $mask_on \
    --pretrained_model=../output/model_iter179999 \
    --image_path=../dataset/coco/val2017/  \
    --image_name=000000000139.jpg \
    --draw_threshold=0.6
