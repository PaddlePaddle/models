CUDA_VISIBLE_DEVICES=0 python infer.py --use_gpu 1\  
									--batch_size 64\  
									--cpu_num 2 \
									--model_dir ./model_dir \
									--test_data_path ./test_data\  
									--vocab_path ./vocab_size.txt 