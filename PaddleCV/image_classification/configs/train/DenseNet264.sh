#DenseNet264
#Training details
python train.py \
       --model=DenseNet264 \
       --batch_size=256 \
       --total_images=1281167 \
       --image_shape=3,224,224 \
       --class_dim=1000 \
       --lr_strategy=piecewise_decay \
       --lr=0.1 \
       --num_epochs=120 \
       --model_save_dir=output/ \
       --l2_decay=1e-4
