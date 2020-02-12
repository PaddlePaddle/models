#!/bin/bash
set -ux

SAVE_DIR=outputs/DSTC7_AVSD.infer
VOCAB_PATH=model/Bert/vocab.txt
DATA_DIR=data/DSTC7_AVSD
INIT_CHECKPOINT=outputs/DSTC7_AVSD/best.model
DATA_TYPE=multi_knowledge

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
    --batch_size 4 \
    --num_type_embeddings 3 \
    --use_discriminator true \
    --init_checkpoint $INIT_CHECKPOINT \
    --save_dir $SAVE_DIR
