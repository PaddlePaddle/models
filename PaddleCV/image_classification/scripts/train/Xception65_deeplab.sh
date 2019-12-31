#Xception65_deeplab
python train.py \
       --model=Xception65_deeplab \
       --batch_size=256 \
       --image_shape 3 299 299 \
       --lr_strategy=cosine_decay \
       --lr=0.045 \
       --num_epochs=120 \
       --model_save_dir=output/ \
       --l2_decay=1e-4 \
       --resize_short_size=320
