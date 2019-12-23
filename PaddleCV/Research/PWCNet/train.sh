#!/usr/bin/env bash
python3 train.py --dataset FlyingChairs --train_val_txt data_dir/FlyingChairs_release/FlyingChairs_train_val.txt --data_root data_dir/FlyingChairs_release/data
# use multi gpus NEED TO DO LATER
#python3 -m paddle.distributed.launch --selected_gpus=0,1 --log_dir ./mylog train.py --use_multi_gpu --batch_size 20 --dataset FlyingChairs --train_val_txt data_dir/FlyingChairs_release/FlyingChairs_train_val.txt --data_root data_dir/FlyingChairs_release/data
