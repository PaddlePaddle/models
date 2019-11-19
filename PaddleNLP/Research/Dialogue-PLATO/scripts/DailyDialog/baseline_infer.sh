#!/bin/bash
set -ux

SAVE_DIR=outputs/DailyDialog.baseline.infer
VOCAB_PATH=model/Bert/vocab.txt
DATA_DIR=data/DailyDialog
INIT_CHECKPOINT=outputs/DailyDialog.baseline/best.model
DATA_TYPE=multi

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=0

# Paddle environment settings.
export FLAGS_fraction_of_gpu_memory_to_use=0.1
export FLAGS_eager_delete_scope=True
export FLAGS_eager_delete_tensor_gb=0.0

python -u \
    ./preprocess.py \
    --vocab_path $VOCAB_PATH \
    --data_dir $DATA_DIR \
    --data_type $DATA_TYPE

python -u \
    ./run.py \
    --do_infer true \
    --vocab_path $VOCAB_PATH \
    --data_dir $DATA_DIR \
    --data_type $DATA_TYPE \
    --batch_size 48 \
    --num_latent 0 \
    --num_type_embeddings 2 \
    --init_checkpoint $INIT_CHECKPOINT \
    --length_average true \
    --save_dir $SAVE_DIR
