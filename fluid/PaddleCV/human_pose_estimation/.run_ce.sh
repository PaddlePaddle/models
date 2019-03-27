#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${human_pose_estimation:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --dataset=coco --num_epochs=3 --batch_num=50 --enable_ce | python _ce.py


cudaid=${human_pose_estimation_4:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --dataset=coco --num_epochs=3 --batch_num=50 --enable_ce | python _ce.py

