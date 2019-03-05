#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


export CPU_NUM=1
export NUM_THREADS=1

FLAGS_benchmark=true  python train.py --enable_ce --train_dir train_big_data/ --vocab_text_path big_vocab_text.txt --vocab_tag_path big_vocab_tag.txt --model_dir big_model --batch_size 500 | python _ce.py

cudaid=${tagspace:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --enable_ce --use_cuda 1 --train_dir train_big_data/ --vocab_text_path big_vocab_text.txt --vocab_tag_path big_vocab_tag.txt --model_dir big_model --batch_size 500 --parallel 1 | python _ce.py

cudaid=${tagspace_4:=0,1,2,3} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --enable_ce --use_cuda 1 --train_dir train_big_data/ --vocab_text_path big_vocab_text.txt --vocab_tag_path big_vocab_tag.txt --model_dir big_model --batch_size 500 --parallel 1 | python _ce.py
