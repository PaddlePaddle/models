#!/bin/bash
source ./env/env.sh
source ./env/utils.sh
source ./env/cloud_job_conf.conf

iplist=$1
#iplist=`echo $nodelist | xargs  | sed 's/ /,/g'`

if [ ! -d log ]
then
    mkdir log
fi

export GLOG_vmodule=fuse_all_reduce_op_pass=10,alloc_continuous_space_for_grad_pass=10

if [[ ${FUSE} == "1" ]]; then
    export FLAGS_fuse_parameter_memory_size=64 #MB
fi

set -ux
check_iplist

distributed_args=""
if [[ ${NUM_CARDS} == "1" ]]; then
    distributed_args="--selected_gpus 0"
fi

node_ips=${PADDLE_TRAINERS}

distributed_args="--node_ips ${PADDLE_TRAINERS} --node_id ${PADDLE_TRAINER_ID} --current_node_ip ${POD_IP} --nproc_per_node 8 --selected_gpus 0,1,2,3,4,5,6,7"
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_RETRY_CNT=10
export FLAGS_sync_nccl_allreduce=0

BATCH_SIZE=1250
python -u ./src/launch.py ${distributed_args} \
	./src/train.py \
	--src_vocab_size 37007 \
	--tgt_vocab_size 37007 \
	--train_file_pattern 'data/translate-train-*' \
	--token_delimiter ' ' \
	--batch_size ${BATCH_SIZE} \
	--use_py_reader True \
	--use_delay_load True \
    --nccl_comm_num ${NCCL_COMM_NUM} \
    --use_hierarchical_allreduce ${USE_HIERARCHICAL_ALLREDUCE} \
	--fetch_steps 50 \
    --fuse ${FUSE} \
    --val_file_pattern 'testset/testfile' \
    --infer_batch_size 32 \
    --decode_alpha 0.3 \
    --beam_size 4 \
    --use_fp16 True \
	learning_rate 2.0 \
	warmup_steps 8000 \
	beta2 0.997 \
	d_model 1024 \
	d_inner_hid 4096 \
	n_head 16 \
	prepostprocess_dropout 0.3 \
	attention_dropout 0.1 \
	relu_dropout 0.1 \
	embedding_sharing True \
	pass_num 100 \
	max_length 256 \
    save_freq 5000 \
	model_dir 'output'

