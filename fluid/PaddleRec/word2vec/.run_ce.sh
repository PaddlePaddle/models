#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


export CPU_NUM=1

FLAGS_benchmark=true  python train.py --train_data_path ./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled --dict_path data/1-billion_dict --with_hs --is_local --num_passes 10 --enable_ce | python _ce.py

export CPU_NUM=8

FLAGS_benchmark=true  python train.py --train_data_path ./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled --dict_path data/1-billion_dict --with_hs --is_local --num_passes 10 --enable_ce | python _ce.py
