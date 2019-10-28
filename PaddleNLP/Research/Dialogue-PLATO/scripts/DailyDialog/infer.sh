#!/bin/bash
set -ux

SAVE_DIR=outputs/DailyDialog.infer
VOCAB_PATH=data/vocab.txt
DATA_DIR=data/DailyDialog
INIT_CHECKPOINT=outputs/DailyDialog/best.model
DATA_TYPE=multi

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=0

# Paddle environment settings.
export FLAGS_fraction_of_gpu_memory_to_use=0.1
export FLAGS_eager_delete_scope=True
export FLAGS_eager_delete_tensor_gb=0.0

if [[ ! -e $DATA_DIR/dial.test.jsonl ]]; then
    python -u \
        ./preprocess.py \
        --vocab_path $VOCAB_PATH \
        --data_dir $DATA_DIR \
        --data_type $DATA_TYPE
fi

python -u \
    ./run.py \
    --do_infer true \
    --vocab_path $VOCAB_PATH \
    --data_dir $DATA_DIR \
    --data_type $DATA_TYPE \
    --batch_size 2 \
    --num_type_embeddings 2 \
    --use_discriminator true \
    --init_checkpoint $INIT_CHECKPOINT \
    --save_dir $SAVE_DIR
