#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

mode=$1 # gpu, cpu, mkldnn
if [ "$mode" = "CPU" ]; then
  use_gpu="False"
  model_path="cpu_model"
elif [ "$mode" = "GPU" ]; then
  use_gpu="True"
  model_path="gpu_model"
elif [ "$mode" = "MKLDNN" ]; then
  use_gpu="False"
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
    --model_path $model_path/model_00001 \
    --input_images_list ~/.cache/paddle/dataset/ctc_data/data/test.list \
    --input_images_dir ~/.cache/paddle/dataset/ctc_data/data/test_images \
    --use_gpu $use_gpu \
    --batch_size 32 \
    --iterations 5 \
    --skip_batch_num 2
