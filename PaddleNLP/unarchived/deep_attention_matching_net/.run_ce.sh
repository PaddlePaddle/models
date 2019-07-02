###!/bin/bash
####This file is only used for continuous evaluation.

export CE_MODE_X=1
export CUDA_VISIBLE_DEVICES=0
export FLAGS_eager_delete_tensor_gb=0.0
if  [ ! -e data_small.pkl ]; then
    wget -c http://dam-data.bj.bcebos.com/data_small.pkl
fi

python train_and_evaluate.py  --data_path data_small.pkl \
                              --use_cuda \
                              --use_pyreader \
                              --num_scan_data 1 \
                              --batch_size 100 | python _ce.py
