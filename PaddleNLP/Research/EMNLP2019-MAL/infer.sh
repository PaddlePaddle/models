#! /bin/sh

export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64:/home/work/cudnn/cudnn_v7/cuda/lib64:/home/work/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

wget https://baidu-nlp.bj.bcebos.com/EMNLP2019-MAL/checkpoint.best.tgz
tar -zxf checkpoint.best.tgz

infer(){
        CUDA_VISIBLE_DEVICES=$1 python -u src/infer.py \
          --val_file_pattern $3 \
          --vocab_size $4 \
          --special_token '<s>' '<e>' '<unk>' \
          --use_mem_opt True \
          --use_delay_load True \
          --infer_batch_size 16 \
          --decode_alpha 0.3 \
          d_model 1024 \
          d_inner_hid 4096 \
          n_head 16 \
          prepostprocess_dropout 0.0 \
          attention_dropout 0.0 \
          relu_dropout 0.0 \
          model_path $2 \
          beam_size 4 \
          max_out_len 306 \
          max_length 256
}

infer 0 checkpoint.best testset/testfile 37007

sh evaluate.sh trans/forward_checkpoint.best
grep "BLEU_cased" trans/*
