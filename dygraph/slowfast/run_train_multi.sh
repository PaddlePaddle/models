export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

start_time=$(date +%s)

python3.7 -m paddle.distributed.launch --log_dir=logs \
          train.py \
          --config=slowfast.yaml \
          --use_gpu=True \
          --use_data_parallel=1 \

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "8 card bs=64, max_epoch=239, epoch_factor=1.0,  warmup_epoch=34, all 400 class, preciseBN 200 iter, valid 10 epoch: build kernel time is $(($cost_time/60))min $(($cost_time%60))s"
