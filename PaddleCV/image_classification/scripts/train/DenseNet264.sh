#DenseNet264
#Training details
python train.py \
       --model=DenseNet264 \
       --batch_size=256 \
       --lr_strategy=piecewise_decay \
       --lr=0.1 \
       --num_epochs=120 \
       --model_save_dir=output/ \
       --l2_decay=1e-4
