#!/bin/bash -ex

#Training details
export FLAGS_conv_workspace_size_limit=4000 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

DATA_DIR="Your image dataset path, e.g. ./data/ILSVRC2012/"
DATA_FORMAT="NHWC"
USE_AMP=true #whether to use amp
USE_DALI=true
USE_ADDTO=true

if ${USE_ADDTO} ;then
    export FLAGS_max_inplace_grad_add=8
fi
if ${USE_DALI}; then
    export FLAGS_fraction_of_gpu_memory_to_use=0.8
fi

python train.py \
       --model=ResNet200_vd \
       --data_dir=${DATA_DIR} \
       --batch_size=64 \
       --num_epochs=200 \
       --total_images=1281167 \
       --image_shape 4 224 224 \
       --class_dim=1000 \
       --print_step=10 \
       --model_save_dir=output/ \
       --lr_strategy=cosine_decay \
       --use_amp=${USE_AMP} \
       --scale_loss=128.0 \
       --use_dynamic_loss_scaling=true \
       --data_format=${DATA_FORMAT} \
       --fuse_elewise_add_act_ops=true \
       --fuse_bn_act_ops=true \
       --fuse_bn_add_act_ops=true \
       --enable_addto=${USE_ADDTO} \
       --validate=true \
       --is_profiler=false \
       --profiler_path=profile/ \
       --reader_thread=10 \
       --reader_buf_size=4000 \
       --use_dali=${USE_DALI} \
       --lr=0.1 \
       --l2_decay=1e-4 \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1
