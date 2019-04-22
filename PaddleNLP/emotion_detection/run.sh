#!/bin/bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=3
export FLAGS_fraction_of_gpu_memory_to_use=0.95
TASK_NAME='emotion_detection'
DATA_PATH=./data/
VOCAB_PATH=./data/vocab.txt
CKPT_PATH=./save_models/textcnn
MODEL_PATH=./models/textcnn

# run_train on train.tsv and do_val on dev.tsv
train() {
    python run_classifier.py \
        --task_name ${TASK_NAME} \
        --use_cuda false \
        --do_train true \
        --do_val true \
        --batch_size 64 \
        --data_dir ${DATA_PATH} \
        --vocab_path ${VOCAB_PATH} \
        --output_dir ${CKPT_PATH} \
        --save_steps 200 \
        --validation_steps 200 \
        --epoch 5 \
        --lr 0.002 \
        --config_path ./config.json \
        --skip_steps 200
}
# run_eval on test.tsv
evaluate() {
    python run_classifier.py \
        --task_name ${TASK_NAME} \
        --use_cuda false \
        --do_val true \
        --batch_size 128 \
        --data_dir ${DATA_PATH} \
        --vocab_path ${VOCAB_PATH} \
        --init_checkpoint ${MODEL_PATH} \
        --config_path ./config.json
}
# run_infer on infer.tsv
infer() {
    python run_classifier.py \
        --task_name ${TASK_NAME} \
        --use_cuda false \
        --do_infer true \
        --batch_size 32 \
        --data_dir ${DATA_PATH} \
        --vocab_path ${VOCAB_PATH} \
        --init_checkpoint ${MODEL_PATH} \
        --config_path ./config.json
}

main() {
    local cmd=${1:-help}
    case "${cmd}" in
        train)
            train "$@";
            ;;
        eval)
            evaluate "$@";
            ;;
        infer)
            infer "$@";
            ;;
        help)
            echo "Usage: ${BASH_SOURCE} {train|eval|infer}";
            return 0;
            ;;
        *)
            echo "unsupport command [${cmd}]";
            echo "Usage: ${BASH_SOURCE} {train|eval|infer}";
            return 1;
            ;;
    esac
}
main "$@"
