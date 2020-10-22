#!/bin/bash -ex

export FLAGS_conv_workspace_size_limit=4000 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

DATA_DIR="Your image dataset path, e.g. /work/datasets/ILSVRC2012/"

DATA_FORMAT="NHWC"
USE_FP16=true #whether to use float16
USE_DALI=true

if ${USE_DALI}; then
    export FLAGS_fraction_of_gpu_memory_to_use=0.8
fi

python train.py \
       --model=ResNet50 \
       --data_dir=${DATA_DIR} \
       --batch_size=256 \
       --total_images=1281167 \
       --image_shape 3 224 224 \
       --class_dim=1000 \
       --print_step=10 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --use_fp16=${USE_FP16} \
       --scale_loss=128.0 \
       --use_dynamic_loss_scaling=true \
       --data_format=${DATA_FORMAT} \
       --fuse_elewise_add_act_ops=true \
       --fuse_bn_act_ops=true \
       --validate=true \
       --is_profiler=false \
       --profiler_path=profile/ \
       --reader_thread=10 \
       --reader_buf_size=4000 \
       --use_dali=${USE_DALI} \
       --lr=0.1

