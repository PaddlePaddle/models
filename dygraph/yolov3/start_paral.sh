python -m paddle.distributed.launch --selected_gpu=0,1,2,3 --started_port=9999 train.py --batch_size=16 --use_data_parallel=1
