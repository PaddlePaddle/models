python train.py --use_gpu False\  #是否使用gpu
			    --epochs 100\  #训练轮次
			    --batch_size 64\  #batch_size大小
			    --embed_size 12\  #每个featsigns的embedding维度
			    --cpu_num 2\  #cpu数量
			    --model_dir ./model_dir \  #模型保存路径
			    --train_data_path ./train_data \  #训练数据路径
			    --vocab_path ./vocab/vocab_size.txt #embedding词汇表大小路径