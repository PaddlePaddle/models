#Training details
#GPU: NVIDIA® Tesla® V100 8cards 200epochs 367h
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#InceptionV4
python train.py \
	    --model=InceptionV4 \
            --batch_size=256 \
            --image_shape 3 299 299 \
            --lr_strategy=cosine_decay \
            --lr=0.045 \
            --num_epochs=200 \
            --model_save_dir=output/ \
            --l2_decay=1e-4 \
            --use_mixup=True \
            --resize_short_size=320 \
            --use_label_smoothing=True \
            --label_smoothing_epsilon=0.1 \
