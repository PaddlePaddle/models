export PATH=~/python_fleet/bin:$PATH
DATA_DIR=/ssd2/lilong/gen_data

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1

python -m paddle.distributed.launch \
    --cluster_node_ips=127.0.0.1 --node_ip=127.0.0.1 --selected_gpus="6,7" --log_dir=mylog \
    train.py \
    --src_vocab_fpath ${DATA_DIR}/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
    --trg_vocab_fpath ${DATA_DIR}/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
    --special_token '<s>' '<e>' '<unk>' \
    --train_file_pattern ${DATA_DIR}/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
    --token_delimiter ' ' \
    --use_token_batch True \
    --batch_size 2048 \
    --sort_type pool \
    --pool_size 200000 \
    --fuse True \
    n_head 8 \
    d_model 512 \
    d_inner_hid 2048 \
    prepostprocess_dropout 0.3
