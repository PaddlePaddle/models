set -eux

export FLAGS_fraction_of_gpu_memory_to_use=0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_selected_gpus=0
export CUDA_VISIBLE_DEVICES=0

# train
python run_ernie_sequence_labeling.py \
    --ernie_config_path "../LARK/ERNIE/config/ernie_config.json" \
    --checkpoints "./checkpoints" \
    --init_pretraining_params "pretrained/params/" \
    --epoch 10 \
    --save_steps 1000 \
    --validation_steps 1000 \
    --lr 2e-4 \
    --crf_learning_rate 0.2 \
    --init_bound 0.1 \
    --skip_steps 1 \
    --vocab_path "../LARK/ERNIE/config/vocab.txt" \
    --batch_size 32 \
    --random_seed 0 \
    --num_labels 57 \
    --max_seq_len 64 \
    --train_set "./data/train.tsv" \
    --test_set "./data/test.tsv" \
    --label_map_config "./conf/label_map.json" \
    --do_lower_case true \
    --use_cuda true \
    --do_train true \
    --do_test true
