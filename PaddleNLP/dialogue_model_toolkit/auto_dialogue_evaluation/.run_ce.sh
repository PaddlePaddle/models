#!/bin/sh
export CE_MODE_X=ce
export FLAGS_eager_delete_tensor_gb=0.0

export CUDA_VISIBLE_DEVICES=0

python -u main.py \
  --do_train True \
    --use_cuda \
      --save_path model_files_tmp/matching_pretrained \
        --train_path data/unlabel_data/train.ids \
          --val_path data/unlabel_data/val.ids \
          --print_step 3 \
          --num_scan_data 3 | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u main.py \
  --do_train True \
    --use_cuda \
      --save_path model_files_tmp/matching_pretrained \
        --train_path data/unlabel_data/train.ids \
          --val_path data/unlabel_data/val.ids \
          --print_step 3 \
          --num_scan_data 3 | python _ce.py
