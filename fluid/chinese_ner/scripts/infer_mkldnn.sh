#!/bin/bash
FLAGS_use_mkldnn=1 time python ../infer.py \
	--device CPU \
	--num_passes 100 \
	--skip_pass_num 2 \
	--profile \
	--test_data_dir ../data/test_files \
	--test_label_file ../data/label_dict \
	--model_path mkldnn_model/params_pass_0

