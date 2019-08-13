export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=0,5,6,7
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#AlexNet:
python train.py \
       --model=AlexNet \
       --batch_size=256 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --with_mem_opt=True \
       --lr_strategy=piecewise_decay \
	--num_epochs=120 \
       --lr=0.01 \
       --l2_decay=1e-4
