#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

mode=$1 # gpu, cpu, mkldnn
if [ "$mode" = "CPU" ]; then
  device="CPU"
  model_path="cpu_model"
elif [ "$mode" = "GPU" ]; then
  device="GPU"
  model_path="gpu_model"
elif [ "$mode" = "MKLDNN" ]; then
  device="CPU"
  model_path="mkldnn_model"
  export FLAGS_use_mkldnn=1
else
  echo "Invalid mode provided. Please use one of {GPU, CPU, MKLDNN}"
  exit 1
fi

ht=`lscpu |grep "per core"|awk -F':' '{print $2}'|xargs`
if [ $ht -eq 1 ]; then # HT is OFF
    if [ -z "$KMP_AFFINITY" ]; then
        export KMP_AFFINITY="granularity=fine,compact,0,0"
    fi
    if [ -z "$OMP_DYNAMIC" ]; then
        export OMP_DYNAMIC="FALSE"
    fi
else # HT is ON
    if [ -z "$KMP_AFFINITY" ]; then
        export KMP_AFFINITY="granularity=fine,compact,1,0"
    fi
fi

python ../infer.py \
	--device $device \
	--num_passes 1 \
	--skip_pass_num 2 \
	--profile \
	--test_data_dir ../data/test_files \
	--test_label_file ../data/label_dict \
	--model_path $model_path/params_pass_0
