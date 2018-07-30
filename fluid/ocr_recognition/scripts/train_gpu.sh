#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/usr/local/lib

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

python ../ctc_train.py \
    --use_gpu True \
    --parallel True \
    --batch_size 32 \
    --save_model_period 1 \
    --total_step 1 \
    --save_model_dir gpu_model