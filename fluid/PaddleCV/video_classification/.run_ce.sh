#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${video_classification:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

export FLAGS_fraction_of_gpu_memory_to_use=0.5
FLAGS_benchmark=true  python train.py --batch_size=16 --total_videos=9537 --class_dim=101 --num_epochs=1 --image_shape=3,224,224 --model_save_dir=output/ --with_mem_opt=True --lr_init=0.01 --num_layers=50 --seg_num=7 --enable_ce=True | python _ce.py
#export FLAGS_fraction_of_gpu_memory_to_use=0.92


cudaid=${video_classification_4:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --batch_size=16 --total_videos=9537 --class_dim=101 --num_epochs=1 --image_shape=3,224,224 --model_save_dir=output/ --with_mem_opt=True --lr_init=0.01 --num_layers=50 --seg_num=7 --enable_ce=True | python _ce.py

exit 0

cudaid=${video_classification_8:=0,1,2,3,4,5,6,7} # use 0,1,2,3,4,5,6,7 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --batch_size=16 --total_videos=9537 --class_dim=101 --num_epochs=1 --image_shape=3,224,224 --model_save_dir=output/ --with_mem_opt=True --lr_init=0.01 --num_layers=50 --seg_num=7 --enable_ce=True | python _ce.py
