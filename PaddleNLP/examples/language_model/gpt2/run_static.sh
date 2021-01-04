export PYTHONPATH=../../../
unset CUDA_VISIBLE_DEVICES
#fleetrun --gpus 0,1,2,3 run_pretrain_static.py --model_name_or_path gpt2-medium-en --input_dir "./input_data"\
fleetrun --gpus 5,6 run_pretrain_static.py --model_name_or_path gpt2-medium-en --input_dir "./input_data"\
    --output_dir "output"\
    --use_recompute true\
    --learning_rate 0.00015\
    --weight_decay 0.01\
    --max_steps 1000\
    --warmup_rate .1\
    --batch_size 8
