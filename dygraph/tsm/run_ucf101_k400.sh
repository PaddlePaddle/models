CUDA_VISIBLE_DEVICES=4,5,6,7 python3.7 -m paddle.distributed.launch --started_port 38989 --log_dir ./mylog.ucf101.frames.k400  train.py --config=./tsm_ucf101.yaml --use_gpu=True --use_data_parallel=True --weights=k400_wei/TSM.pdparams 
