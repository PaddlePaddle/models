export CUDA_VISIBLE_DEVICES=4,5,6,7
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

export FLAGS_conv_workspace_size_limit=4000 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1


DATA_DIR="./data/ILSVRC2012/"

DATA_FORMAT="NCHW"
USE_FP16=true #whether to use float16
USE_DALI=true
USE_ADDTO=true

if ${USE_FP16} ;then
    DATA_FORMAT="NHWC"
fi

if ${USE_ADDTO} ;then
    export FLAGS_max_inplace_grad_add=8
fi

if ${USE_DALI}; then
    export FLAGS_fraction_of_gpu_memory_to_use=0.8
fi


#InceptionV3
python train.py \
	    --model=InceptionV3 \
            --batch_size=512 \
            --image_shape 3 299 299 \
            --lr_strategy=cosine_decay \
            --lr=0.09 \
            --num_epochs=100 \
            --model_save_dir=output/ \
            --l2_decay=1e-4 \
            --use_mixup=False \
            --resize_short_size=320 \
            --use_label_smoothing=True \
            --label_smoothing_epsilon=0.1 \
            --use_fp16=${USE_FP16} \
            --scale_loss=128.0 \
            --use_dynamic_loss_scaling=true \
            --data_format=${DATA_FORMAT} \
            --fuse_elewise_add_act_ops=true \
            --fuse_bn_act_ops=true \
            --fuse_bn_add_act_ops=true \
            --use_dali=${USE_DALI} \
            --enable_addto=${USE_ADDTO} \
