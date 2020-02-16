#Training details
#Missed
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#ResNeXt101_32x4d
python train.py \
	--model=ResNeXt101_32x4d \
        --batch_size=256 \
        --lr_strategy=piecewise_decay \
        --lr=0.1 \
        --num_epochs=120 \
        --model_save_dir=output/ \
        --l2_decay=1e-4
