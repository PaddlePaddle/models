#!/bin/bash
time python ../infer.py \
	--device CPU \
	--num_passes 100 \
	--skip_pass_num 2 \
	--profile \
	--test_data_dir ../data/test_files \
	--test_label_file ../data/label_dict \
	--model_path model_mkldnn/params_pass_0

