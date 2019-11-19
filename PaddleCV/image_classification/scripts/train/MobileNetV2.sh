#Training details
#GPU: NVIDIA® Tesla® V100 4cards 240epochs 135h
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98


python train.py \
	--model=MobileNetV2 \
	--batch_size=500 \
	--model_save_dir=output/ \
	--lr_strategy=cosine_decay \
	--num_epochs=240 \
	--lr=0.1 \
	--l2_decay=4e-5
