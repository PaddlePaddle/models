#MobileNetV2_x2_0
python train.py \
       --model=MobileNetV2_x2_0 \
       --batch_size=256 \
       --model_save_dir=output/ \
       --lr_strategy=cosine_decay \
       --num_epochs=240 \
       --lr=0.1 \
       --l2_decay=4e-5 
