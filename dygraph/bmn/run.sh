export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch \
       --selected_gpus=0,1,2,3 \
       --log_dir ./mylog \
       train.py --use_data_parallel True
