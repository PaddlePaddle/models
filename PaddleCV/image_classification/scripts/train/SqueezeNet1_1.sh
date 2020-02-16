#SqueezeNet1_1
python train.py \
        --model=SqueezeNet1_1 \
        --batch_size=256 \
        --lr_strategy=cosine_decay \
        --model_save_dir=output/ \
        --lr=0.02 \
        --num_epochs=120 \
        --l2_decay=1e-4
