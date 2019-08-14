# python train.py \
#       --model=DenseNet121 \
#       --batch_size=256 \
#       --total_images=1281167 \
#       --image_shape=3,224,224 \
#       --class_dim=1000 \
#       --lr_strategy=piecewise_decay \
#       --lr=0.1 \
#       --num_epochs=120 \
#       --with_mem_opt=True \
#       --model_save_dir=output/ \
#       --l2_decay=1e-4
