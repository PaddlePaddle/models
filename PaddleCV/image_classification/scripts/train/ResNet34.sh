#Training details
#GPU: NVIDIA® Tesla® V100 4cards 120epochs 73h
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
#ResNet34:
python train.py \
	--model=ResNet34 \
	--batch_size=256 \
	--model_save_dir=output/ \
	--lr_strategy=cosine_decay \
	--lr=0.1 \
	--num_epochs=120 \
	--l2_decay=1e-4
