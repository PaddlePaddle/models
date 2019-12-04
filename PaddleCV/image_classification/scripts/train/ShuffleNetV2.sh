##Training details
#GPU: NVIDIA® Tesla® K40 4cards 240epochs 156h
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

python train.py \
	--model=ShuffleNetV2 \
	--batch_size=1024 \
	--model_save_dir=output/ \
	--lr_strategy=cosine_decay_warmup \
	--lr=0.5 \
	--num_epochs=240 \
	--l2_decay=4e-5 
