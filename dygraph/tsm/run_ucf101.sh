CUDA_VISIBLE_DEVICES=0,1,2,3 python3.7 -m paddle.distributed.launch --started_port 38989 --log_dir ./mylog.ucf101.frames  tsm.py --config=./tsm_ucf101.yaml --use_gpu=True --use_data_parallel=True  
