#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1



cudaid=${HiNAS_models:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train_hinas.py --model=0 --random_flip_left_right=False --random_flip_up_down=False --pad_and_cut_image=False --shuffle_image=False --batch_size=128 --num_epochs=1 --cutout=False --dropout_rate=0.5 --enable_ce=True| python _ce.py


cudaid=${HiNAS_models_4:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train_hinas.py --model=0 --random_flip_left_right=False --random_flip_up_down=False --pad_and_cut_image=False --shuffle_image=False --batch_size=128 --num_epochs=1 --cutout=False --dropout_rate=0.5 --enable_ce=True| python _ce.py


cudaid=${HiNAS_models_8:=0,1,2,3,4,5,6,7} # use 0,1,2,3,4,5,6,7 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train_hinas.py --model=0 --random_flip_left_right=False --random_flip_up_down=False --pad_and_cut_image=False --shuffle_image=False --batch_size=128 --num_epochs=1 --cutout=False --dropout_rate=0.5 --enable_ce=True| python _ce.py

