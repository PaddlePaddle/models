export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../../../
python run_pretrain.py --model_name_or_path gpt2-small-en --input_dir "./input_data"\
    --output_dir "output"\
    --max_lr 0.00015\
    --min_lr 0.00001\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --batch_size 8\
    --device gpu
