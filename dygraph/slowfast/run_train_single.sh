export CUDA_VISIBLE_DEVICES=1

start_time=$(date +%s)

python3.7 train.py --config=slowfast-single.yaml --use_gpu=True --use_data_parallel=0

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "1 card bs=4,  max_epoch=10, epoch_factor=1.5,  warmup_epoch=1, part 10 class, preciseBN 200 iter, use decord, build kernel time is $(($cost_time/60))min $(($cost_time%60))s"
