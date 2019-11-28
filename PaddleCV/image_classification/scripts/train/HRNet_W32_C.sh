#Training details
#HRNet_W32_C
python train.py \
       --model=HRNet_W32_C \
       --batch_size=256 \
       --total_images=1281167 \
       --class_dim=1000 \
       --lr_strategy=piecewise_decay \
       --lr=0.1 \
       --num_epochs=120 \
       --model_save_dir=output/ \
       --l2_decay=1e-4
