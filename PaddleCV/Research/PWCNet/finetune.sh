#!/usr/bin/env bash
python3 train.py --loss l1 --pretrained ./out/pwc_net_paddle --dataset FlyingChairs --train_val_txt data_dir/FlyingChairs_release/FlyingChairs_train_val.txt --data_root data_dir/FlyingChairs_release/data

# use multi gpus NEED TO DO LATER
#python3 -m paddle.distributed.launch --selected_gpus=0,1 train.py --use_multi_gpu --batch_size 40 --loss l1 --pretrained ./out/pwc_net_paddle --dataset FlyingChairs --train_val_txt data_dir/FlyingChairs_release/FlyingChairs_train_val.txt --data_root data_dir/FlyingChairs_release/data
