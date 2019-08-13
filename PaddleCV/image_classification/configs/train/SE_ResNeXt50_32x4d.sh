#SE_ResNeXt50_32x4d:
#python train.py \
#       --model=SE_ResNeXt50_32x4d \
#       --batch_size=400 \
#       --total_images=1281167 \
#	--class_dim=1000 \
#       --image_shape=3,224,224 \
#       --lr_strategy=cosine_decay \
#       --model_save_dir=output/ \
#       --lr=0.1 \
#       --num_epochs=200 \
#       --with_mem_opt=True \
#       --l2_decay=1.2e-4
