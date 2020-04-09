python share_bottom.py  --use_gpu False\  #使用cpu训练
						--train_path data/data24913/train_data/\  #训练数据路径
						--test_path data/data24913/test_data/\  #测试数据路径
						--batch_size 32\  #设置batch_size大小
						--feature_size 499\  #设置特征维度
						--bottom_size 117\  #设置bottom网络大小
						--tower_nums 2\  #设置tower数量
						--tower_size 8\  #设置tower网络大小
						--epochs 400  #设置epoch轮次