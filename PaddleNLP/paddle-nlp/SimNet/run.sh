#!/usr/bin/env bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=3
export FLAGS_fraction_of_gpu_memory_to_use=0.95
TASK_NAME='simnet'
TRAIN_DATA_PATH=./data/lcqmc.train.pair
VALID_DATA_PATH=./data/lcqmc.dev
TEST_DATA_PATH=./data/lcqmc.test
INFER_DATA_PATH=./data/infer_data
VOCAB_PATH=./term2id.dict
CKPT_PATH=./save_models
TEST_RESULT_PATH=./test_result
INFER_RESULT_PATH=./infer_result
TASK_MODE='pairwise'
CONFIG_PATH=./config/bow_pairwise.json
INIT_CHECKPOINT=./save_models/model


# run_train
train() {
	python run_classifier.py \
		--task_name ${TASK_NAME} \
		--use_cuda false \
		--do_train True \
		--do_valid True \
		--do_test True \
		--do_infer False \
		--batch_size 128 \
		--train_data_dir ${TRAIN_DATA_PATH} \
		--valid_data_dir ${VALID_DATA_PATH} \
		--test_data_dir ${TEST_DATA_PATH} \
		--infer_data_dir ${INFER_DATA_PATH} \
		--output_dir ${CKPT_PATH} \
		--config_path ${CONFIG_PATH} \
		--vocab_path ${VOCAB_PATH} \
		--epoch 10 \
		--save_steps 1000 \
		--validation_steps 100 \
		--task_mode ${TASK_MODE}
}
#run_test
test() {
	python run_classifier.py \
		--task_name ${TASK_NAME} \
		--use_cuda false \
		--do_test True \
		--verbose_result True \
		--batch_size 128 \
		--test_data_dir ${TEST_DATA_PATH} \
		--test_result_path ${TEST_RESULT_PATH} \
		--config_path ${CONFIG_PATH} \
		--vocab_path ${VOCAB_PATH} \
		--task_mode ${TASK_MODE} \
		--init_checkpoint ${INIT_CHECKPOINT}
}
# run_eval
infer() {
	python run_classifier.py \
		--task_name ${TASK_NAME} \
		--use_cuda false \
		--do_infer True \
		--batch_size 128 \
		--infer_data_dir ${INFER_DATA_PATH} \
		--infer_result_path ${INFER_RESULT_PATH} \
		--config_path ${CONFIG_PATH} \
		--vocab_path ${VOCAB_PATH} \
		--task_mode ${TASK_MODE} \
		--init_checkpoint ${INIT_CHECKPOINT}
}

$1