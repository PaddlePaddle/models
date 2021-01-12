export PYTHONPATH=../../../
unset CUDA_VISIBLE_DEVICES
#fleetrun --gpus 0,1,2,3 run_pretrain_static.py --model_name_or_path gpt2-medium-en --input_dir "./input_data"\
export FLAGS_call_stack_level=2
fleetrun --gpus 0,1,2,3,4,5,6,7 run_pretrain_static.py --model_name_or_path gpt2-small-en --input_dir "./input_data/final_dataset"\
    --output_dir "output"\
    --learning_rate 0.00015\
    --weight_decay 0.01\
    --save_steps 2000\
    --max_steps 320000\
    --warmup_rate .01\
    --batch_size 32\
    --grad_clip 1.0\
    --logging_steps 1\
    --scale_loss 1024\
    --use_amp True\

