#SE_ResNet50_vd

export FLAGS_conv_workspace_size_limit=4000 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

DATA_DIR="/paddle/dataset/ILSVRC2012/"

# DATA_FORMAT="NCHW"
# # DATA_FORMAT="NHWC"
# USE_FP16=false #whether to use float16
# USE_DALI=false
# USE_ADDTO=false

DATA_FORMAT="NCHW"
# DATA_FORMAT="NHWC"
USE_FP16=true #whether to use float16
USE_DALI=true
USE_ADDTO=true

if ${USE_ADDTO} ;then
    export FLAGS_max_inplace_grad_add=8
fi

if ${USE_DALI}; then
    export FLAGS_fraction_of_gpu_memory_to_use=0.8
fi

python3.6 train.py \
       --model=SE_ResNet50_vd \
       --data_dir=${DATA_DIR} \
       --batch_size=128 \
       --lr_strategy=cosine_decay \
       --use_fp16=${USE_FP16} \
       --data_format=${DATA_FORMAT} \
       --lr=0.1 \
       --num_epochs=200 \
       --model_save_dir=output/ \
       --l2_decay=1e-4 \
       --use_mixup=False \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1 \
       --enable_addto=${USE_ADDTO} \
       --use_dali=${USE_DALI} \
       --image_shape 4 224 224 \
       --fuse_bn_act_ops=true \
       --fuse_bn_add_act_ops=true \
