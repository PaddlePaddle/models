#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${face_detection:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --dataset=coco --num_epochs=2 --batch_num=10 --enable_ce | python _ce.py


cudaid=${face_detection_4:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --dataset=coco --num_epochs=2 --batch_num=10 --enable_ce | python _ce.py


cudaid=${face_detection_8:=0,1,2,3,4,5,6,7} # use 0,1,2,3,4,5,6,7 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --dataset=coco --num_epochs=2 --batch_num=10 --enable_ce | python _ce.py


