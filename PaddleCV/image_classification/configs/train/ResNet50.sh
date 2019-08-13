#ResNet50:
#python train.py \
#       --model=ResNet50 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=piecewise_decay \
#	--num_epochs=120 \
#       --lr=0.1 \
#       --l2_decay=1e-4
