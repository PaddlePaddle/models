#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

function run_train() {
    echo "training"
    python run_sequence_labeling.py \
        --do_train True \
        --do_test True \
        --do_infer False \
        --batch_size 100 \
        --traindata_dir ./data/train_data \
        --testdata_dir ./data/test_data \
        --model_save_dir ./models \
        --save_model_per_batches 10000 \
        --eval_window 20 \
        --batch_size 100 \
        --use_gpu 0 \
        --traindata_shuffle_buffer 200000 \
        --word_emb_dim 128 \
        --grnn_hidden_dim 256 \
        --bigru_num 2 \
        --base_learning_rate 1e-3 \
        --emb_learning_rate 5 \
        --crf_learning_rate 0.2 \
        --word_dict_path ./conf/word.dic \
        --label_dict_path ./conf/tag.dic \
        --word_rep_dict_path ./conf/q2b.dic \
        --num_iterations 0
}

function run_infer() {
    echo "infering"
    python run_sequence_labeling.py \
        --do_train False \
        --do_test False \
        --do_infer True \
        --batch_size 80 \
        --model_path ./model/ \
        --testdata_dir ./data/test_data \
        --word_dict_path ./conf/word.dic \
        --label_dict_path ./conf/tag.dic \
        --word_rep_dict_path ./conf/q2b.dic
}

function main() {
    local cmd=${1:-help}
    case "${cmd}" in
        train)
            run_train "$@";
            ;;
        infer)
            run_infer "$@";
            ;;
        help)
            echo "Usage: ${BASH_SOURCE} {train|infer}";
            return 0;
            ;;
        *)
            echo "unsupport command [${cmd}]";
            echo "Usage: ${BASH_SOURCE} {train|infer}";
            return 1;
            ;;
    esac
}

main "$@"
