###!/bin/bash
####This file is only used for continuous evaluation.

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

if [ ! -d "/root/.cache/paddle/dataset/pascalvoc" ];then
    mkdir -p /root/.cache/paddle/dataset/pascalvoc
    ./data/pascalvoc/download.sh
    cp -r ./data/pascalvoc/. /home/.cache/paddle/dataset/pascalvoc
fi

cudaid=${object_detection_cudaid:=0}
export CUDA_VISIBLE_DEVICES=$cudaid
FLAGS_benchmark=true  python train.py --enable_ce=True --batch_size=64 --epoc_num=2 --data_dir=/root/.cache/paddle/dataset/pascalvoc/ | python _ce.py

cudaid=${object_detection_cudaid_m:=0,1,2,3}
export CUDA_VISIBLE_DEVICES=$cudaid
FLAGS_benchmark=true  python train.py --enable_ce=True --batch_size=64 --epoc_num=2 --data_dir=/root/.cache/paddle/dataset/pascalvoc/ | python _ce.py
