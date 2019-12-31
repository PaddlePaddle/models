#SE_ResNet34_vd
python train.py \
       --model=SE_ResNet34_vd \
       --batch_size=256 \
       --total_images=1281167 \
       --class_dim=1000 \
       --lr_strategy=cosine_decay \
       --lr=0.1 \
       --num_epochs=200 \
       --model_save_dir=output/ \
       --l2_decay=7e-5 \
       --use_mixup=True \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1 \
