#!/bin/bash
# This file is only used for continuous evaluation.

export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=3

if [ ! -d 'pretrain' ]; then
    ln -s ${PRETRAINED_MODEL_PATH} ./pretrain
fi

if [ ! -d 'data' ]; then
    ln -s ${ILSVRC2012_DATA_PATH} ./data
fi

if [ -d 'checkpoints' ]; then
    rm -rf checkpoints
fi

sed -i "s/epoch: 200/epoch: 1/g" configs/filter_pruning_uniform.yaml

python compress.py \
    --model "MobileNet" \
    --pretrained_model ./pretrain/MobileNetV1_pretrained \
    --compress_config ./configs/filter_pruning_uniform.yaml 2>&1 | tee run.log | python _ce.py



