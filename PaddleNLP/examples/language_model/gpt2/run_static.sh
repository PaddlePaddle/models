export PYTHONPATH=../../../
unset CUDA_VISIBLE_DEVICES
#fleetrun --gpus 0,1,2,3 run_pretrain_static.py --model_name_or_path gpt2-medium-en --input_dir "./input_data"\
export FLAGS_call_stack_level=2
fleetrun --gpus 5 --log_dir ./new_log run_pretrain_static.py --model_name_or_path gpt2-small-en --input_dir "./new_data"\
    --output_dir "output"\
    --max_lr 0.00015\
    --min_lr 0.00001\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --max_steps 100000\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --batch_size 8\
