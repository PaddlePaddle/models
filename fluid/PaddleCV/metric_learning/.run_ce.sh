#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${metric_learning:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train_elem.py  --model=ResNet50 --train_batch_size=80 --test_batch_size=80  --lr=0.01 --total_iter_num=10 --use_gpu=True --model_save_dir=out_put --loss_name=arcmargin --arc_scale=80.0  --arc_margin=0.15  --arc_easy_margin=False --enable_ce=True | python _ce.py


cudaid=${metric_learning_4:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train_elem.py  --model=ResNet50 --train_batch_size=80 --test_batch_size=80  --lr=0.01 --total_iter_num=10 --use_gpu=True --model_save_dir=out_put --loss_name=arcmargin --arc_scale=80.0  --arc_margin=0.15  --arc_easy_margin=False --enable_ce=True | python _ce.py


cudaid=${metric_learning_8:=0,1,2,3,4,5,6,7} # use 0,1,2,3,4,5,6,7 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train_elem.py  --model=ResNet50 --train_batch_size=80 --test_batch_size=80  --lr=0.01 --total_iter_num=10 --use_gpu=True --model_save_dir=out_put --loss_name=arcmargin --arc_scale=80.0  --arc_margin=0.15  --arc_easy_margin=False --enable_ce=True | python _ce.py

