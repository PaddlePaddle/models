#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

batch_size=32
core_num=`lscpu |grep -m1 "CPU(s)"|awk -F':' '{print $2}'|xargs`
mode=$1 # gpu, cpu, mkldnn
if [ "$mode" = "CPU" ]; then
  if [ $core_num -gt $batch_size ]; then
    echo "Batch size should be greater or equal to the number of 
          available cores, when parallel mode is set to True."
  fi
  use_gpu="False"
  save_model_dir="cpu_model"
  parallel="True"
elif [ "$mode" = "GPU" ]; then
  use_gpu="True"
  save_model_dir="gpu_model"
  parallel="True"
elif [ "$mode" = "MKLDNN" ]; then
  use_gpu="False"
  save_model_dir="mkldnn_model"
  parallel="False"
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
    --use_gpu $use_gpu \
    --parallel $parallel \
    --batch_size $batch_size \
    --save_model_period 1 \
    --total_step 1 \
    --save_model_dir $save_model_dir

