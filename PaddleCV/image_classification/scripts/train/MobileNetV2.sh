#Training details
#GPU: NVIDIA® Tesla® V100 4cards 240epochs 135h
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fraction_of_gpu_memory_to_use=0.98


python train.py \
	--model=MobileNetV2 \
	--batch_size=500 \
	--total_images=1281167 \
	--class_dim=1000 \
	--image_shape=3,224,224 \
	--model_save_dir=output/ \
	--lr_strategy=cosine_decay \
	--num_epochs=240 \
	--lr=0.1 \
	--l2_decay=4e-5
