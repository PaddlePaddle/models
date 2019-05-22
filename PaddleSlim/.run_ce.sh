#!/bin/bash
# This file is only used for continuous evaluation.

export CUDA_VISIBLE_DEVICES=0

export LD_LIBRARY_PATH=/home/work/cuda-9.2/lib64:/home/users/dangqingqing/nccl2/nccl_2.1.15-1+cuda9.0_x86_64/lib:/home/work/cudnn/cudnn_v7.1/cuda/lib64:/home/liuxudong/.jumbo/opt/gcc53/lib64:$LD_LIBRARY_PATH

ln -s ${PRETRAINED_MODEL_PATH} ./pretrain
ln -s ${ILSVRC2012_DATA_PATH} ./data

sed -i "s/epoch: 200/epoch: 5/g" configs/filter_pruning_uniform.yaml

python compress.py \
    --model "MobileNet" \
    --pretrained_model ./pretrain/MobileNetV1_pretrained \
    --compress_config ./configs/filter_pruning_uniform.yaml | python _ce.py



