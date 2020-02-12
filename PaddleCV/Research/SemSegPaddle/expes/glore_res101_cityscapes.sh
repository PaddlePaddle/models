#!/bin/bash


# GloRe_Res101_Cityscapes
# 1.1 Training
CUDA_VISIBLE_DEVICES=0,1,2,3  python train.py  --use_gpu \
                                              --use_mpio \
                                              --cfg ./configs/glore_res101_cityscapes.yaml | tee -a train.log 2>&1
# 1.2 single-scale testing
CUDA_VISIBLE_DEVICES=0 python eval.py --use_gpu \
                                      --cfg ./configs/glore_res101_cityscapes.yaml
# 1.3 multi-scale testing
CUDA_VISIBLE_DEVICES=0 python eval.py --use_gpu \
                                      --multi_scales \
                                      --cfg ./configs/glore_res101_cityscapes.yaml

