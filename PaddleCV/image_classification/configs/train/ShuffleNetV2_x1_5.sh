#python train.py \
#       --model=ShuffleNetV2_x1_5 \
#       --batch_size=512 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --with_mem_opt=True \
#       --lr_strategy=cosine_warmup_decay \
#       --num_epochs=240 \
#       --lr=0.25 \
#       --l2_decay=4e-5 \
#       --lower_ratio=1.0 \
#       --upper_ratio=1.0
