#SqueezeNet1_1
python train.py \
        --model=SqueezeNet1_1 \
        --batch_size=256 \
        --total_images=1281167 \
        --image_shape=3,224,224 \
        --lr_strategy=cosine_decay \
        --class_dim=1000 \
        --model_save_dir=output/ \
        --lr=0.02 \
        --num_epochs=120 \
        --l2_decay=1e-4
