python run_pretrain.py --model_name_or_path gpt2-medium-en --input_dir "./input_data"\
    --output_dir "output"\
    --learning_rate 0.00015\
    --weight_decay 0.01\
    --max_steps 50\
    --warmup_rate .1\
    --batch_size 2
