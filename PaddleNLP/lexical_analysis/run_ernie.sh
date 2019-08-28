#set -eux
#export FLAGS_fraction_of_gpu_memory_to_use=0.02
export FLAGS_eager_delete_tensor_gb=1.0
#export FLAGS_fast_eager_deletion_mode=1
export FLAGS_sync_nccl_allreduce=1
#export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
#export GLOG_v=1
#export GLOG_logtostderr=1
#export FLAGS_selected_gpus=0        # which GPU to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

ERNIE_PRETRAINED_MODEL_PATH=./pretrained/
ERNIE_FINETUNED_MODEL_PATH=./ernie_models/step_10
DATA_PATH=./data/
source /home/work/huangdingbang/.bash_fluid
# train
function run_train() {
    echo "training"
    fluid run_ernie_sequence_labeling.py \
        --mode train \
        --ernie_config_path "${ERNIE_PRETRAINED_MODEL_PATH}/ernie_config.json" \
        --model_save_dir "./ernie_models" \
        --init_pretraining_params "${ERNIE_PRETRAINED_MODEL_PATH}/params/" \
        --epoch 10 \
        --save_steps 5 \
        --validation_steps 5 \
        --base_learning_rate 2e-4 \
        --crf_learning_rate 0.2 \
        --init_bound 0.1 \
        --print_steps 1 \
        --vocab_path "${ERNIE_PRETRAINED_MODEL_PATH}/vocab.txt" \
        --batch_size 64 \
        --random_seed 0 \
        --num_labels 57 \
        --max_seq_len 128 \
        --train_data "${DATA_PATH}/train2.tsv" \
        --test_data "${DATA_PATH}/test2.tsv" \
        --label_map_config "./conf/label_map.json" \
        --do_lower_case true \
        --use_cuda true \
        --cpu_num 4
}


function run_eval() {
    echo "evaluating"
    fluid run_ernie_sequence_labeling.py \
        --mode eval \
        --ernie_config_path "${ERNIE_PRETRAINED_MODEL_PATH}/ernie_config.json" \
        --init_checkpoint "${ERNIE_FINETUNED_MODEL_PATH}" \
        --init_bound 0.1 \
        --vocab_path "${ERNIE_PRETRAINED_MODEL_PATH}/vocab.txt" \
        --batch_size 64 \
        --random_seed 0 \
        --num_labels 57 \
        --max_seq_len 128 \
        --test_data "${DATA_PATH}/test2.tsv" \
        --label_map_config "./conf/label_map.json" \
        --do_lower_case true \
        --use_cuda true

}

function run_infer() {
    echo "infering"
    fluid run_ernie_sequence_labeling.py \
        --mode infer \
        --ernie_config_path "${ERNIE_PRETRAINED_MODEL_PATH}/ernie_config.json" \
        --init_checkpoint "${ERNIE_FINETUNED_MODEL_PATH}" \
        --init_bound 0.1 \
        --vocab_path "${ERNIE_PRETRAINED_MODEL_PATH}/vocab.txt" \
        --batch_size 64 \
        --random_seed 0 \
        --num_labels 57 \
        --max_seq_len 128 \
        --test_data "${DATA_PATH}/test2.tsv" \
        --label_map_config "./conf/label_map.json" \
        --do_lower_case true \
        --use_cuda true

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
