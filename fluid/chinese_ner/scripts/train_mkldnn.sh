#!/bin/bash
FLAGS_use_mkldnn=1 time python ../train.py \
	--device CPU \
	--model_save_dir mkldnn_model \
	--test_data_dir ../data/test_files \
	--train_data_dir ../data/train_files \
	--num_passes 1
