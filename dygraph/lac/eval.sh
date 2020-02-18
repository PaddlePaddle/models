#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

python eval.py     --batch_size 200     --word_emb_dim 128     --grnn_hidden_dim 128     --bigru_num 2     --use_cuda False     --init_checkpoint ./padding_models/step_120000    --test_data ./data/test.tsv     --word_dict_path ./conf/word.dic     --label_dict_path ./conf/tag.dic     --word_rep_dict_path ./conf/q2b.dic

