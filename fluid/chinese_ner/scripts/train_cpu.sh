#!/bin/bash
time python ../train.py \
	--device CPU \
	--model_save_dir cpu_model \
	--test_data_dir ../data/test_files \
	--train_data_dir ../data/train_files \
	--num_passes 1
