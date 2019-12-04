#Training details
#GPU: NVIDIA® Tesla® V100 4cards 150epochs 173h
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#VGG19:
python train.py \
	--model=VGG19 \
	--batch_size=256 \
	--lr_strategy=cosine_decay \
	--lr=0.01 \
	--num_epochs=150 \
        --model_save_dir=output/ \
	--l2_decay=4e-4
