#!/usr/bin/env bash
model_files_path="./model_files"

#get pretrained_bow_pairwise_model

wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/simnet_bow_pairwise_dygraph.tar.gz
if [ ! -d $model_files_path ]; then
	mkdir $model_files_path
fi
tar xzf simnet_bow_pairwise_dygraph.tar.gz -C $model_files_path
rm simnet_bow_pairwise_dygraph.tar.gz
