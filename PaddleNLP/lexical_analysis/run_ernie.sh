set -eux

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_sync_nccl_allreduce=1
export FLAGS_selected_gpus=0        # which GPU to use
export CUDA_VISIBLE_DEVICES=0

ERNIE_PRETRAINED_MODEL_PATH=./pretrained/
ERNIE_FINETUNED_MODEL_PATH=./model_finetuned/
DATA_PATH=./data/

# train
function run_train() {
    echo "training"
    python run_ernie_sequence_labeling.py \
        --ernie_config_path "${ERNIE_PRETRAINED_MODEL_PATH}/ernie_config.json" \
        --checkpoints "./checkpoints" \
        --init_pretraining_params "${ERNIE_PRETRAINED_MODEL_PATH}/params/" \
        --epoch 10 \
        --save_steps 1000 \
        --validation_steps 1000 \
        --lr 2e-4 \
        --crf_learning_rate 0.2 \
        --init_bound 0.1 \
        --skip_steps 1 \
        --vocab_path "${ERNIE_PRETRAINED_MODEL_PATH}/vocab.txt" \
        --batch_size 64 \
        --random_seed 0 \
        --num_labels 57 \
        --max_seq_len 128 \
        --train_set "${DATA_PATH}/train.tsv" \
        --test_set "${DATA_PATH}/test.tsv" \
        --label_map_config "./conf/label_map.json" \
        --do_lower_case true \
        --use_cuda false \
        --do_train true \
        --do_test true \
        --do_infer false
}


function run_eval() {
    echo "evaluating"
    python run_ernie_sequence_labeling.py \
        --ernie_config_path "${ERNIE_PRETRAINED_MODEL_PATH}/ernie_config.json" \
        --init_pretraining_params "${ERNIE_PRETRAINED_MODEL_PATH}/params/" \
        --init_checkpoint "${ERNIE_FINETUNED_MODEL_PATH}" \
        --init_bound 0.1 \
        --vocab_path "${ERNIE_PRETRAINED_MODEL_PATH}/vocab.txt" \
        --batch_size 64 \
        --random_seed 0 \
        --num_labels 57 \
        --max_seq_len 128 \
        --test_set "${DATA_PATH}/test.tsv" \
        --label_map_config "./conf/label_map.json" \
        --do_lower_case true \
        --use_cuda true \
        --do_train false \
        --do_test true \
        --do_infer false
}


function run_infer() {
    echo "infering"
    python run_ernie_sequence_labeling.py \
        --ernie_config_path "${ERNIE_PRETRAINED_MODEL_PATH}/ernie_config.json" \
        --init_pretraining_params "${ERNIE_PRETRAINED_MODEL_PATH}/params/" \
        --init_checkpoint "${ERNIE_FINETUNED_MODEL_PATH}" \
        --init_bound 0.1 \
        --vocab_path "${ERNIE_PRETRAINED_MODEL_PATH}/vocab.txt" \
        --batch_size 64 \
        --random_seed 0 \
        --num_labels 57 \
        --max_seq_len 128 \
        --infer_set "${DATA_PATH}/test.tsv" \
        --label_map_config "./conf/label_map.json" \
        --do_lower_case true \
        --use_cuda true \
        --do_train false \
        --do_test false \
        --do_infer true
}


function main() {
    local cmd=${1:-help}
    case "${cmd}" in
        train)
            run_train "$@";
            ;;
        eval)
            run_eval "$@";
            ;;
        infer)
            run_infer "$@";
            ;;
        help)
            echo "Usage: ${BASH_SOURCE} {train|test|infer}";
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
