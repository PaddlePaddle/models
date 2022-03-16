#!/bin/bash

function func_parser_value(){
    strs=$1
    IFS=$2
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

function status_check(){
    last_status=$1   # the exit code
    run_command=$2
    run_log=$3
    if [ $last_status -eq 0 ]; then
        echo -e "\033[33m Run successfully with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    else
        echo -e "\033[33m Run failed with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    fi
}


IFS=$'\n'

BASIC_CONFIG="config.txt"
basic_dataline=$(cat $BASIC_CONFIG)
basic_lines=(${basic_dataline})

TIPC_CONFIG=$1
tipc_dataline=$(cat $TIPC_CONFIG)
tipc_lines=(${tipc_dataline})


# parser basic config
label_path=$(func_parser_value "${basic_lines[1]}" " ")
resize_short_size=$(func_parser_value "${basic_lines[2]}" " ")
crop_size=$(func_parser_value "${basic_lines[3]}" " ")
visualize=$(func_parser_value "${basic_lines[4]}" " ")
enable_benchmark=$(func_parser_value "${basic_lines[9]}" " ")
tipc_benchmark=$(func_parser_value "${basic_lines[10]}" " ")

# parser tipc config
runtime_device=$(func_parser_value "${tipc_lines[0]}" ":")
lite_arm_work_path=$(func_parser_value "${tipc_lines[1]}" ":")
lite_arm_so_path=$(func_parser_value "${tipc_lines[2]}" ":")
clas_model_name=$(func_parser_value "${tipc_lines[3]}" ":")
inference_cmd=$(func_parser_value "${tipc_lines[4]}" ":")
num_threads_list=$(func_parser_value "${tipc_lines[5]}" ":")
batch_size_list=$(func_parser_value "${tipc_lines[6]}" ":")
precision_list=$(func_parser_value "${tipc_lines[7]}" ":")

LOG_PATH="./output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results.log"


#run Lite TIPC
function func_test_tipc(){
    IFS="|"
    _basic_config=$1
    _model_name=$2
    _log_path=$3
    for num_threads in ${num_threads_list[*]}; do
        sed -i " " "s/num_threads.*/num_threads ${num_threads}/" ${_basic_config}
        for batch_size in ${batch_size_list[*]}; do
            sed -i " " "s/batch_size.*/batch_size ${batch_size}/" ${_basic_config}
            for precision in ${precision_list[*]}; do
                sed -i " " "s/precision.*/precision ${precision}/" ${_basic_config}
                _save_log_path="${_log_path}/lite_${_model_name}_runtime_device_${runtime_device}_precision_${precision}_batchsize_${batch_size}_threads_${num_threads}.log"
                real_inference_cmd=$(echo ${inference_cmd} | awk -F " " '{print path $1" "path $2" "path $3}' path="$lite_arm_work_path")
                command1="adb push ${_basic_config} ${lite_arm_work_path}"
                eval ${command1}
                command2="adb shell 'export LD_LIBRARY_PATH=${lite_arm_work_path}; ${real_inference_cmd}'  > ${_save_log_path} 2>&1"
                eval ${command2}
                status_check $? "${command2}" "${status_log}"
            done
        done
    done
}


echo "################### run test tipc ###################"
sed -i " " "s/runtime_device.*/runtime_device arm_cpu/" ${BASIC_CONFIG}
escape_lite_arm_work_path=$(echo ${lite_arm_work_path//\//\\\/})
sed -i " " "s/clas_model_file.*/clas_model_file  ${escape_lite_arm_work_path}${clas_model_name}/" ${BASIC_CONFIG}
func_test_tipc ${BASIC_CONFIG} ${clas_model_name} ${LOG_PATH}
