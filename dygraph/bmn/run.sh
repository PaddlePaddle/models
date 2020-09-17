export CUDA_VISIBLE_DEVICES=0,1,2,3

start_time=$(date +%s)

python3 train.py --use_data_parallel=1

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "4 card bs=16 9 epoch training time is $(($cost_time/60))min $(($cost_time%60))s"
