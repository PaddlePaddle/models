python train.py \
       --model=ShuffleNetV2_x1_5 \
       --batch_size=512 \
       --model_save_dir=output/ \
       --lr_strategy=cosine_decay_warmup \
       --num_epochs=240 \
       --lr=0.25 \
       --l2_decay=4e-5 \
       --lower_ratio=1.0 \
       --upper_ratio=1.0
