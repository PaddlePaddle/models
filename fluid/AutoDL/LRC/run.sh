export LD_LIBRARY_PATH=/usr/lib64:/usr/local/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 python -u train_mixup.py \
--batch_size=80 \
--auxiliary \
--weight_decay=0.0003 \
--learning_rate=0.025 \
--rcc_loss_lambda=0.7 \
--cutout

