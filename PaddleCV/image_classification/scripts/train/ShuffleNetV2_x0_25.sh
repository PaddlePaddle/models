#ShuffleNetV2_x0_25:
python train.py \
       --model=ShuffleNetV2_x0_25 \
       --batch_size=1024 \
       --model_save_dir=output/ \
       --lr_strategy=cosine_decay_warmup \
       --num_epochs=240 \
       --lr=0.5 \
       --l2_decay=3e-5 \
       --lower_scale=0.64 \
       --lower_ratio=0.8 \
       --upper_ratio=1.2
