python train_mmoe.py  --use_gpu False							#使用cpu训练
					  --train_path data/data24913/train_data/	#训练数据路径
					  --test_path data/data24913/test_data/		#测试数据路径
					  --batch_size 32							#设置batch_size大小
					  --expert_num 8							#设置expert数量
					  --gate_num 2								#设置gate数量
					  --epochs 400								#设置epoch轮次