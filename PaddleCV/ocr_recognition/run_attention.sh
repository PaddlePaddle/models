export CUDA_VISIBLE_DEVICES=0
nohup python train.py \
--lr=1.0 \
--gradient_clip=5.0 \
--model="attention" \
--log_period=10 \
> attention.log 2>&1 &

tailf attention.log
