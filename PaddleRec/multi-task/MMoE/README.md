# MMOE

 以下是本例的简要目录结构及说明： 

```
├── README.md            # 文档
├── requirements.txt     # 需要的安装包
├── mmoe_train.py        # mmoe模型脚本
├── utils                # 通用函数
├── args                 # 参数脚本
├── create_data.sh       # 生成训练数据脚本
├── data_preparation.py  # 数据预处理脚本
├── train_gpu.sh		 # gpu训练脚本
├── train_cpu.sh		 # cpu训练脚本
```

## 简介

多任务模型通过学习不同任务的联系和差异，可提高每个任务的学习效率和质量。多任务学习的的框架广泛采用shared-bottom的结构，不同任务间共用底部的隐层。这种结构本质上可以减少过拟合的风险，但是效果上可能受到任务差异和数据分布带来的影响。  论文[《Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts》]( https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture- )中提出了一个Multi-gate Mixture-of-Experts(MMOE)的多任务学习结构。MMOE模型刻画了任务相关性，基于共享表示来学习特定任务的函数，避免了明显增加参数的缺点。 

我们在Paddlepaddle定义MMOE的网络结构，在开源数据集Census-income Data上验证模型效果，两个任务的auc分别为：

1.income

> max_mmoe_test_auc_income：0.94937
>
> mean_mmoe_test_auc_income：0.94465

2.marital

> max_mmoe_test_auc_marital：0.99419
>
> mean_mmoe_test_auc_marital：0.99324

本项目支持GPU和CPU两种单机训练环境。



## 数据下载及预处理

数据地址： [Census-income Data](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz )

数据解压后， 在create_data.sh脚本文件中添加文件的路径，并运行脚本。

```sh
mkdir train_data
mkdir test_data
mkdir data
train_path="data/census-income.data"
test_path="data/census-income.test"
train_data_path="train_data/"
test_data_path="test_data/"
<<<<<<< HEAD
pip install -r requirements.txt
=======

>>>>>>> 282e48904fbd6168835966b4e0c7851c82d46e23
wget -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz
tar -zxvf data/census.tar.gz -C data/

python data_preparation.py --train_path ${train_path} \
                           --test_path ${test_path} \
                           --train_data_path ${train_data_path}\
                           --test_data_path ${test_data_path}

```

## 环境

 PaddlePaddle 1.7.0 

 python3.7 

## 单机训练

GPU环境

在train_gpu.sh脚本文件中设置好数据路径、参数。

```sh
CUDA_VISIBLE_DEVICES=0 python train_mmoe.py  --use_gpu 1 \  #使用gpu训练
<<<<<<< HEAD
                                    --train_data_path 'train_data'\  #训练数据路径
                                    --test_data_path 'test_data'\  #测试数据路径
                                    --model_dir 'model_dir'\  #模型保存地址
                                    --feature_size 499\  #设置特征的维度
                                    --batch_size 32\  #设置batch_size大小
                                    --expert_num 8\  #设置expert数量
                                    --gate_num 2\  #设置gate数量
                                    --expert_size 16\  #设置expert网络大小
                                    --tower_size 8\  #设置tower网络大小
                                    --epochs 400 #设置epoch轮次
=======
                      --train_path data/data24913/train_data/\  #训练数据路径
                      --test_path data/data24913/test_data/\  #测试数据路径
                      --feature_size 499\  #设置特征的维度
                      --batch_size 32\  #设置batch_size大小
                      --expert_num 8\  #设置expert数量
                      --gate_num 2\  #设置gate数量
                      --expert_size 16\  #设置expert网络大小
                      --tower_size 8\  #设置tower网络大小
                      --epochs 400 #设置epoch轮次
>>>>>>> 282e48904fbd6168835966b4e0c7851c82d46e23
```

修改脚本的可执行权限并运行

```
./train_gpu.sh
```

CPU环境

在train_cpu.sh脚本文件中设置好数据路径、参数。

```sh
python train_mmoe.py  --use_gpu 0 \  #使用cpu训练
<<<<<<< HEAD
                    --train_data_path 'train_data'\  #训练数据路径
                    --test_data_path 'test_data'\  #测试数据路径
                    --model_dir 'model_dir'\  #模型保存地址
                    --feature_size 499\  #设置特征的维度
                    --batch_size 32\  #设置batch_size大小
                    --expert_num 8\  #设置expert数量
                    --gate_num 2\  #设置gate数量
                    --expert_size 16\  #设置expert网络大小
                    --tower_size 8\  #设置tower网络大小
                    --epochs 400 #设置epoch轮次
=======
                      --train_path data/data24913/train_data/\  #训练数据路径
                      --test_path data/data24913/test_data/\  #测试数据路径
                      --feature_size 499\  #设置特征的维度
                      --batch_size 32\  #设置batch_size大小
                      --expert_num 8\  #设置expert数量
                      --gate_num 2\  #设置gate数量
                      --expert_size 16\  #设置expert网络大小
                      --tower_size 8\  #设置tower网络大小
                      --epochs 400 #设置epoch轮次
>>>>>>> 282e48904fbd6168835966b4e0c7851c82d46e23
```

修改脚本的可执行权限并运行

```
./train_cpu.sh
```



## 预测

本模型训练和预测交替进行，运行train_mmoe.py 即可得到预测结果

## 模型效果

epoch设置为100的训练和测试效果如下：

```text
batch_size:[32],feature_size:[499],expert_num:[8],gate_num[2],expert_size[16],tower_size[8],epochs:[100]
2020-04-16 11:28:06,- INFO - epoch_id: 0,epoch_time: 129.17434 s,loss: 0.62215,train_auc_income: 0.86302,train_auc_marital: 0.92316,test_auc_income: 0.84525,test_auc_marital: 0.98269
2020-04-16 11:30:36,- INFO - epoch_id: 1,epoch_time: 149.79017 s,loss: 0.42484,train_auc_income: 0.90634,train_auc_marital: 0.98418,test_auc_income: 
......
2020-04-16 15:31:23,- INFO - epoch_id: 97,epoch_time: 147.07304 s,loss: 0.30267,train_auc_income: 0.94743,train_auc_marital: 0.99430,test_auc_income: 0.94905,test_auc_marital: 0.99414
2020-04-16 15:33:51,- INFO - epoch_id: 98,epoch_time: 148.34412 s,loss: 0.29688,train_auc_income: 0.94736,train_auc_marital: 0.99433,test_auc_income: 0.94846,test_auc_marital: 0.99409
2020-04-16 15:36:21,- INFO - epoch_id: 99,epoch_time: 149.91047 s,loss: 0.31330,train_auc_income: 0.94732,train_auc_marital: 0.99403,test_auc_income: 0.94881,test_auc_marital: 0.99386
2020-04-16 15:36:21,- INFO - mean_mmoe_test_auc_income: 0.94465,mean_mmoe_test_auc_marital 0.99324,max_mmoe_test_auc_income: 0.94937,max_mmoe_test_auc_marital 0.99419
```
