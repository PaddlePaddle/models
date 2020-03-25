# MMOE

 以下是本例的简要目录结构及说明：

```python
├── README.md            # 文档
├── train_mmoe.py        # mmoe模型脚本
├── utils                # 通用函数
├── args                 # 参数脚本
├── create_data.sh       # 生成训练数据脚本
├── train_path           # 原始训练数据文件
├── test_path            # 原始测试数据文件
├── train_data_path      # 训练数据路径
└── test_data_path       # 测试数据路径
```

## 简介

 		多任务模型通过学习不同任务的联系和差异，可提高每个任务的学习效率和质量。多任务学习的的框架广泛采用shared-bottom的结构，不同任务间共用底部的隐层。  论文[MMOE][ https://dl.acm.org/doi/10.1145/3219819.3220007 ]中提出了一个Multi-gate Mixture-of-Experts(MMoE)的多任务学习结构。MMoE模型刻画了任务相关性，基于共享表示来学习特定任务的函数，避免了明显增加参数的缺点。

## 数据下载及预处理

数据地址：https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD

数据解压后， 在create_data.sh脚本文件中添加文件的路径，并运行脚本。

```shell
mkdir data/data24913/train_data 		#新建训练数据目录
mkdir data/data24913/test_data			#新建测试数据目录
mkdir data/data24913/validation_data 	#新建验证数据目录

train_path="data/data24913/census-income.data" 			#原始训练数据路径
test_path="data/data24913/census-income.test" 			#原始测试数据路径
train_data_path="data/data24913/train_data/" 			#处理后训练数据路径
test_data_path="data/data24913/test_data/"				#处理后测试数据路径
validation_data_path="data/data24913/validation_data/"	#处理后验证数据路径

python data_preparation.py --train_path ${train_path} \
                           --test_path ${test_path} \
                           --train_data_path ${train_data_path}\
                           --test_data_path ${test_data_path}\
                           --validation_data_path ${validation_data_path}
```

## 单机训练

GPU环境

```shell
python train_mmoe.py  --use_gpu True
					  --train_path data/data24913/train_data/
					  --test_path data/data24913/test_data/
					  --batch_size 32
					  --expert_num 8
					  --gate_num 2
					  --epochs 400
```



CPU环境

```shell
python train_mmoe.py  --use_gpu False
					  --train_path data/data24913/train_data/
					  --test_path data/data24913/test_data/
					  --batch_size 32
					  --expert_num 8
					  --gate_num 2
					  --epochs 400


```

## 预测

本模型训练和预测交替进行，运行train_mmoe.py 即可得到预测结果

## 模型效果

epoch设置为100的效果如下：

```shell
epoch_id:[0],epoch_time:[136.99230 s],loss:[0.48952],train_auc_income:[0.52317],train_auc_marital:[0.78102],test_auc_income:[0.52329],test_auc_marital:[0.84055]
epoch_id:[1],epoch_time:[137.79457 s],loss:[0.48089],train_auc_income:[0.52466],train_auc_marital:[0.92589],test_auc_income:[0.52842],test_auc_marital:[0.93463]
epoch_id:[2],epoch_time:[137.22369 s],loss:[0.43654],train_auc_income:[0.63070],train_auc_marital:[0.95467],test_auc_income:[0.65807],test_auc_marital:[0.95781]
epoch_id:[3],epoch_time:[133.58558 s],loss:[0.44318],train_auc_income:[0.73284],train_auc_marital:[0.96599],test_auc_income:[0.74561],test_auc_marital:[0.96750]
epoch_id:[4],epoch_time:[128.61714 s],loss:[0.41398],train_auc_income:[0.78572],train_auc_marital:[0.97190],test_auc_income:[0.79312],test_auc_marital:[0.97280]
epoch_id:[5],epoch_time:[126.85907 s],loss:[0.44676],train_auc_income:[0.81760],train_auc_marital:[0.97549],test_auc_income:[0.82190],test_auc_marital:[0.97609]
epoch_id:[6],epoch_time:[131.20426 s],loss:[0.40833],train_auc_income:[0.83818],train_auc_marital:[0.97796],test_auc_income:[0.84132],test_auc_marital:[0.97838]
epoch_id:[7],epoch_time:[130.86647 s],loss:[0.39193],train_auc_income:[0.85259],train_auc_marital:[0.97974],test_auc_income:[0.85512],test_auc_marital:[0.98006]
epoch_id:[8],epoch_time:[137.07437 s],loss:[0.43083],train_auc_income:[0.86343],train_auc_marital:[0.98106],test_auc_income:[0.86520],test_auc_marital:[0.98126]
epoch_id:[9],epoch_time:[138.65452 s],loss:[0.38813],train_auc_income:[0.87173],train_auc_marital:[0.98205],test_auc_income:[0.87317],test_auc_marital:[0.98224]
epoch_id:[10],epoch_time:[135.61756 s],loss:[0.39048],train_auc_income:[0.87839],train_auc_marital:[0.98295],test_auc_income:[0.87954],test_auc_marital:[0.98309]
...
...
epoch_id:[95],epoch_time:[134.57041 s],loss:[0.31102],train_auc_income:[0.93345],train_auc_marital:[0.99191],test_auc_income:[0.93348],test_auc_marital:[0.99192]
epoch_id:[96],epoch_time:[134.19668 s],loss:[0.31128],train_auc_income:[0.93354],train_auc_marital:[0.99193],test_auc_income:[0.93357],test_auc_marital:[0.99193]
epoch_id:[97],epoch_time:[126.89334 s],loss:[0.31202],train_auc_income:[0.93361],train_auc_marital:[0.99195],test_auc_income:[0.93363],test_auc_marital:[0.99195]
epoch_id:[98],epoch_time:[136.01872 s],loss:[0.29857],train_auc_income:[0.93370],train_auc_marital:[0.99197],test_auc_income:[0.93372],test_auc_marital:[0.99197]
epoch_id:[99],epoch_time:[133.60402 s],loss:[0.31113],train_auc_income:[0.93379],train_auc_marital:[0.99199],test_auc_income:[0.93382],test_auc_marital:[0.99199]
```
