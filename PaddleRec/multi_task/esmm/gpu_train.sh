CUDA_VISIBLE_DEVICES=0 python train.py --use_gpu True\  
									   --epochs 100\  
									   --batch_size 64\  
									   --embed_size 12\  
									   --cpu_num 2\  
									   --model_dir './model_dir'\
									   --train_data_path './train_data'\  
									   --vocab_path './vocab/vocab_size.txt' 