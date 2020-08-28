export CUDA_VISIBLE_DEVICES=0

start_time=$(date +%s)

python3.7 train.py --config=slowfast-single.yaml --use_gpu=True --use_data_parallel=0

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "1 card bs=8, 196 epoch 34 warmup epoch, 400 class, preciseBN 200 iter build kernel time is $(($cost_time/60))min $(($cost_time%60))s"
