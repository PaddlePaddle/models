#!/bin/bash


# GloRe_Res101_PascalContext
:<<!
# 1.1 Training
CUDA_VISIBLE_DEVICES=0,1,2,3  python train.py  --use_gpu \
                                              --use_mpio \
                                              --cfg ./configs/glore_res101_pascalcontext.yaml | tee -a train.log 2>&1
!
# 1.2 single-scale testing
CUDA_VISIBLE_DEVICES=0 python eval.py --use_gpu \
                                      --cfg ./configs/glore_res101_pascalcontext.yaml
:<<!
# 1.3 multi-scale testing
CUDA_VISIBLE_DEVICES=0 python eval.py --use_gpu \
                                      --multi_scales \
                                      --cfg ./configs/glore_res101_pascalcontext.yaml

!
