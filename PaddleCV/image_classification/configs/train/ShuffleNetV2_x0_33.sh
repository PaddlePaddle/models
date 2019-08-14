#ShuffleNetV2_x0_33:
#python train.py \
#       --model=ShuffleNetV2_x0_33 \
#       --batch_size=1024 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_warmup_decay \
#       --num_epochs=240 \
#       --lr=0.5 \
#       --l2_decay=3e-5 \
#       --lower_scale=0.64 \
#       --lower_ratio=0.8 \
#       --upper_ratio=1.2
