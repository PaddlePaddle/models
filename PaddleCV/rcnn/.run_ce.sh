#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${face_detection:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --model_save_dir=output/ --data_dir=dataset/coco/ --max_iter=500 --enable_ce --pretrained_model=./imagenet_resnet50_fusebn --learning_rate=0.00125 | python _ce.py


cudaid=${face_detection_m:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --model_save_dir=output/ --data_dir=dataset/coco/ --max_iter=500 --enable_ce --pretrained_model=./imagenet_resnet50_fusebn --learning_rate=0.005 | python _ce.py

