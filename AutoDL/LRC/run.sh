CUDA_VISIBLE_DEVICES=0 python -u train_mixup.py \
--batch_size=80 \
--auxiliary \
--weight_decay=0.0003 \
--learning_rate=0.025 \
--lrc_loss_lambda=0.7 \
--cutout

