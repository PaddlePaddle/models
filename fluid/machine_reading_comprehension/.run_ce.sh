#!/bin/bash

DATA_PATH=./data
if [ ! -e $DATA_PATH/demo ] ; then
    mkdir -p $DATA_PATH/demo
    tar -zxf $DATA_PATH/demo.tgz -C $DATA_PATH/demo
fi

train(){
python  -u run.py   \
	--trainset 'data/demo/search.train.json' \
	--devset 'data/demo/search.dev.json' \
	--testset 'data/demo/search.test.json' \
	--vocab_dir 'data/demo/' \
	--use_gpu true \
	--save_dir ./models \
	--pass_num 1 \
	--learning_rate 0.001 \
	--batch_size 32 \
	--embed_size 300 \
	--hidden_size 150 \
	--max_p_num 5 \
	--max_p_len 500 \
	--max_q_len 60 \
	--max_a_len 200 \
	--drop_rate 0.2 \
	--log_interval 1 \
        --enable_ce \
        --train 
}

cudaid=${transformer:=2} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train | python _ce.py

cudaid=${transformer_m:=2,4} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train | python _ce.py
