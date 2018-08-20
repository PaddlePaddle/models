#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

mode=$1 # gpu, cpu, mkldnn
if [ "$mode" = "CPU" ]; then
  device="CPU"
  parallel="--parallel True"
  save_model_dir="cpu_model"
elif [ "$mode" = "GPU" ]; then
  device="GPU"
  parallel="--parallel True"
  save_model_dir="gpu_model"
elif [ "$mode" = "MKLDNN" ]; then
  device="CPU"
  parallel=""
  save_model_dir="mkldnn_model"
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

python ../train.py \
	--device $device \
  $parallel \
	--model_save_dir $save_model_dir \
	--test_data_dir ../data/test_files \
	--train_data_dir ../data/train_files \
	--num_passes 1
