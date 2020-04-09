python mmoe_train.py  --use_gpu False \  #使用cpu训练
					  --train_data_path data/data24913/train_data/\  #训练数据路径
					  --test_data_path data/data24913/test_data/\  #测试数据路径
					  --feature_size 499\  #设置特征的维度
					  --batch_size 32\  #设置batch_size大小
					  --expert_num 8\  #设置expert数量
					  --gate_num 2\  #设置gate数量
					  --expert_size 16\  #设置expert网络大小
					  --tower_size 8\  #设置tower网络大小
					  --epochs 400 #设置epoch轮次