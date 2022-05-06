#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer', 'whole_infer']
MODE=$2

dataline=$(awk 'NR==1, NR==32{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
gpu_list=$(func_parser_value "${lines[3]}")
train_use_gpu_key=$(func_parser_key "${lines[4]}")
train_use_gpu_value=$(func_parser_value "${lines[4]}")
epoch_key=$(func_parser_key "${lines[5]}")
epoch_num=$(func_parser_params "${lines[5]}" "${MODE}")
save_model_key=$(func_parser_key "${lines[6]}")
train_batch_key=$(func_parser_key "${lines[7]}")
train_batch_value=$(func_parser_params "${lines[7]}" "${MODE}")
pretrain_model_key=$(func_parser_key "${lines[8]}")
pretrain_model_value=$(func_parser_value "${lines[8]}")
train_model_name=$(func_parser_value "${lines[9]}")
data_path_key=$(func_parser_key "${lines[10]}")
data_path_value=$(func_parser_value "${lines[10]}")
# train py
trainer_list=$(func_parser_value "${lines[12]}")
norm_trainer=$(func_parser_key "${lines[13]}")
trainer_py=$(func_parser_value "${lines[13]}")
# nodes
nodes_key=$(func_parser_key "${lines[14]}")
nodes_value=$(func_parser_value "${lines[14]}")

# eval params
eval_py=$(func_parser_value "${lines[16]}")

# infer params
save_infer_key=$(func_parser_key "${lines[19]}")
save_infer_dir=$(func_parser_value "${lines[19]}")
export_weight=$(func_parser_key "${lines[20]}")
norm_export=$(func_parser_value "${lines[21]}")

# parser inference model 
infer_model_dir=$(func_parser_value "${lines[23]}")
infer_export=$(func_parser_value "${lines[24]}")

# parser inference 
inference_py=$(func_parser_value "${lines[26]}")
use_gpu_key=$(func_parser_key "${lines[27]}")
use_gpu_list=$(func_parser_value "${lines[27]}")
batch_size_key=$(func_parser_key "${lines[28]}")
batch_size_list=$(func_parser_value "${lines[28]}")
infer_model_key=$(func_parser_key "${lines[29]}")
image_dir_key=$(func_parser_key "${lines[30]}")
infer_img_dir=$(func_parser_value "${lines[30]}")
benchmark_key=$(func_parser_key "${lines[31]}")
benchmark_value=$(func_parser_value "${lines[31]}")

# log
LOG_PATH="./log/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_python.log"


function func_inference(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    _log_path=$4
    _img_dir=$5
    # inference 
    for use_gpu in ${use_gpu_list[*]}; do
        # cpu
        if [ ${use_gpu} = "False" ] || [ ${use_gpu} = "cpu" ]; then
            for batch_size in ${batch_size_list[*]}; do
                _save_log_path="${_log_path}/python_infer_cpu_batchsize_${batch_size}.log"
                set_infer_data=$(func_set_params "${image_dir_key}" "${_img_dir}")
                set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                set_model_dir=$(func_set_params "${infer_model_key}" "${_model_dir}")
                command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} > ${_save_log_path} 2>&1 "
                eval $command
                last_status=${PIPESTATUS[0]}
                eval "cat ${_save_log_path}"
                status_check $last_status "${command}" "${status_log}"
            done
        # gpu        
        elif [ ${use_gpu} = "True" ] || [ ${use_gpu} = "gpu" ]; then
            for batch_size in ${batch_size_list[*]}; do
                _save_log_path="${_log_path}/python_infer_gpu_batchsize_${batch_size}.log"
                set_infer_data=$(func_set_params "${image_dir_key}" "${_img_dir}")
                set_benchmark=$(func_set_params "${benchmark_key}" "${benchmark_value}")
                set_batchsize=$(func_set_params "${batch_size_key}" "${batch_size}")
                set_model_dir=$(func_set_params "${infer_model_key}" "${_model_dir}")
                command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${set_model_dir} ${set_batchsize} ${set_infer_data} ${set_benchmark} > ${_save_log_path} 2>&1 "
                eval $command
                last_status=${PIPESTATUS[0]}
                eval "cat ${_save_log_path}"
                status_check $last_status "${command}" "${status_log}"   
            done      
        else
            echo "Does not support hardware other than CPU and GPU Currently!"
        fi
    done
}

if [ ${MODE} = "whole_infer" ]; then
    GPUID=$3
    if [ ${#GPUID} -le 0 ];then
        env=" "
    else
        env="export CUDA_VISIBLE_DEVICES=${GPUID}"
    fi
    # set CUDA_VISIBLE_DEVICES
    eval $env
    
    IFS="|"
    
    # run export
    if [ ${infer_export} != "null" ];then
        save_infer_dir="${save_infer_dir}"
        set_export_weight=$(func_set_params "${export_weight}" "${infer_model_dir}")
        set_save_infer_key=$(func_set_params "${save_infer_key}" "${save_infer_dir}")
        export_cmd="${python} ${infer_export} ${set_export_weight} ${set_save_infer_key}"
        echo ${infer_export} 
        echo $export_cmd
        eval $export_cmd
        status_export=$?
        status_check $status_export "${export_cmd}" "${status_log}"
    else
        save_infer_dir=${save_infer_dir}
    fi
    #run inference
    func_inference "${python}" "${inference_py}" "${save_infer_dir}" "${LOG_PATH}" "${infer_img_dir}"
    
else
    IFS="|"
    export Count=0
    USE_GPU_KEY=(${train_use_gpu_value})
    for gpu in ${gpu_list[*]}; do
        train_use_gpu=${USE_GPU_KEY[Count]}
        Count=$(($Count + 1))
        ips=""
        if [ ${gpu} = "-1" ];then
            env=""
        elif [ ${#gpu} -le 1 ];then
            env="export CUDA_VISIBLE_DEVICES=${gpu}"
        elif [ ${#gpu} -le 15 ];then
            IFS=","
            array=(${gpu})
            env="export CUDA_VISIBLE_DEVICES=${array[0]}"
            IFS="|"
        else
            IFS=";"
            array=(${gpu})
            ips=${array[0]}
            gpu=${array[1]}
            IFS=","
            array=(${gpu})
            env="export CUDA_VISIBLE_DEVICES=${array[0]}"
            IFS="|"
        fi

        for trainer in ${trainer_list[*]}; do
        
            run_train=${trainer_py}
            run_export=${norm_export}

            if [ ${run_train} = "null" ]; then
                continue
            fi
            set_epoch=$(func_set_params "${epoch_key}" "${epoch_num}")
            set_pretrain=$(func_set_params "${pretrain_model_key}" "${pretrain_model_value}")
            set_batchsize=$(func_set_params "${train_batch_key}" "${train_batch_value}")
            if [ ${#ips} -le 15 ];then
                save_log="${LOG_PATH}/${trainer}_gpus_${gpu}"
            else                  
                IFS=","
                ips_array=(${ips})
                IFS="|"
                nodes=${#ips_array[@]}
                save_log="${LOG_PATH}/${trainer}_gpus_${gpu}_nodes_${nodes}"
            fi
            set_save_model=$(func_set_params "${save_model_key}" "${save_log}")
            if [ ${#gpu} -le 2 ];then  # train with single gpu
                cmd="${python} ${run_train} ${set_save_model} ${set_epoch} ${set_pretrain} ${set_batchsize}"
            elif [ ${#ips} -le 15 ];then  # train with multi-gpu
                cmd="${python} -m paddle.distributed.launch --gpus=${gpu} ${run_train} ${set_save_model} ${set_epoch} ${set_pretrain} ${set_batchsize}"
            else     # train with multi-machine
                cmd="${python} -m paddle.distributed.launch --ips=${ips} --gpus=${gpu} ${run_train} ${set_save_model} ${set_epoch} ${set_pretrain} ${set_batchsize}"
            fi
            # run train
            eval $cmd
            status_check $? "${cmd}" "${status_log}"
            # TODO ensure model name
            set_eval_pretrain=$(func_set_params "${pretrain_model_key}" "${save_log}/${train_model_name}")

            # run eval 
            if [ ${eval_py} != "null" ]; then
                eval ${env}
                eval_cmd="${python} ${eval_py} ${set_eval_pretrain}" 
                eval $eval_cmd
                status_check $? "${eval_cmd}" "${status_log}"
            fi
            # run export model
            if [ ${run_export} != "null" ]; then 
                # run export model
                save_infer_path="${save_log}"
                set_export_weight=$(func_set_params "${export_weight}" "${save_log}/${train_model_name}")
                set_save_infer_key=$(func_set_params "${save_infer_key}" "${save_infer_path}")
                export_cmd="${python} ${run_export} ${set_export_weight} ${set_save_infer_key}"
                eval $export_cmd
                status_check $? "${export_cmd}" "${status_log}"

                #run inference
                eval $env
                save_infer_path="${save_log}"
                
                infer_model_dir=${save_infer_path}
                
                func_inference "${python}" "${inference_py}" "${infer_model_dir}" "${LOG_PATH}" "${train_infer_img_dir}"
                
                eval "unset CUDA_VISIBLE_DEVICES"
            fi
        done  # done with:    for trainer in ${trainer_list[*]}; do 
    done      # done with:    for gpu in ${gpu_list[*]}; do
fi
