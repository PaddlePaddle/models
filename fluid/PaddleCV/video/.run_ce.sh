#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${video_4:=0,1,2,3} # use 0,1,2,3-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --model-name=TSN --config=./configs/tsn.txt --save-dir=checkpoints --epoch-num=6 --valid-interval=0 --enable_ce | python _ce.py

sleep 10

FLAGS_benchmark=true  python train.py --model-name=AttentionCluster --config=./configs/attention_cluster.txt --save-dir=checkpoints --epoch-num=3  --valid-interval=0 --enable_ce | python _ce.py

sleep 10

cudaid=${video:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --model-name=TSN --config=./configs/tsn.txt --save-dir=checkpoints --epoch-num=6 --valid-interval=0 --enable_ce | python _ce.py

sleep 10

FLAGS_benchmark=true  python train.py --model-name=AttentionCluster --config=./configs/attention_cluster.txt --save-dir=checkpoints --epoch-num=3  --valid-interval=0 --enable_ce | python _ce.py
