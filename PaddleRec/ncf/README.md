# NCF

 以下是本例的简要目录结构及说明： 

```
├── Data/ #原始数据集目录
├── README.md # 文档
├── requirements.txt   # 需要的安装包
├── gmf.py # gmf网络文件
├── mlp.py # mlp网络文件
├── neumf.py # neumf网络文件
├── create_data.sh # 生成训练数据脚本
├── Dataset.py # 测试数据集处理
├── get_train_data.py # 生成测试数据集
├── evaluate.py # 预测并计算指标文件
├── train.py # 训练文件
├── infer.py # 预测文件
├── args.py # 参数文件
├── utils.py # 通用函数
├── train_gpu.sh # gpu训练shell脚本
├── train_cpu.sh # cpu训练shell脚本
```

## 简介

很多应用场景，并没有显性反馈的存在。因为大部分用户是沉默的用户，并不会明确给系统反馈“我对这个物品的偏好值是多少”。因此，推荐系统可以根据大量的隐性反馈来推断用户的偏好值。[《Neural Collaborative Filtering 》](https://arxiv.org/pdf/1708.05031.pdf)作者利用深度学习来对user和item特征进行建模，使模型具有非线性表达能力。具体来说使用多层感知机来学习user-item交互函数，提出了一种隐性反馈协同过滤解决方案。

## 环境

 PaddlePaddle 1.7.0 

 python3.7 

## 单机训练

GPU环境

在train_gpu.sh脚本文件中设置好数据路径、参数。

```sh
CUDA_VISIBLE_DEVICES=0 python train.py --use_gpu 1 \ #使用gpu	
                                        --NeuMF 1 \ #nn和gmf网络结合
                                        --epochs 20 \ #训练轮次
                                        --batch_size 256 \ #batch大小
                                        --num_factors 8 \ #gmf网络输入的embedding大小
                                        --num_neg 4 \ #负采样个数
                                        --lr 0.001 \ #学习率
                                        --model_dir 'model_dir' #模型保存目录
```

修改脚本的可执行权限并运行

```
./train_gpu.sh
```

CPU环境

在train_cpu.sh脚本文件中设置好数据路径、参数。

```sh
python train.py --use_gpu 0 \ #使用cpu	
                --NeuMF 1 \ #nn和gmf网络结合
                --epochs 20 \ #训练轮次
                --batch_size 256 \ #batch大小
                --num_factors 8 \ #gmf网络输入的embedding大小
                --num_neg 4 \ #负采样个数
                --lr 0.001 \ #学习率
                --model_dir 'model_dir'  #模型保存目录
```

修改脚本的可执行权限并运行

```
./train_cpu.sh
```

## 单机预测

预测使用CPU环境，速度较快。

```
python infer.py
```

## 模型效果

训练：

```
use_gpu:1, NeuMF:1, epochs:20, batch_size:256, num_factors:8, num_neg:4, lr:0.001, model_dir:model_dir, layers:[64, 32, 16, 8]
W0428 12:15:20.169631  1161 device_context.cc:237] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.0
W0428 12:15:20.173840  1161 device_context.cc:245] device: 0, cuDNN Version: 7.3.
2020-04-28 12:15:21,945-INFO: epoch: 0, batch_id: 0, batch_time: 0.01069s, loss: 0.69115
2020-04-28 12:15:21,956-INFO: epoch: 0, batch_id: 1, batch_time: 0.00917s, loss: 0.68997
2020-04-28 12:15:21,976-INFO: epoch: 0, batch_id: 2, batch_time: 0.00901s, loss: 0.68813
...
2020-04-28 12:15:22,726-INFO: epoch: 0, batch_id: 72, batch_time: 0.00874s, loss: 0.44167
2020-04-28 12:15:22,736-INFO: epoch: 0, batch_id: 73, batch_time: 0.00862s, loss: 0.44800
2020-04-28 12:15:22,746-INFO: epoch: 0, batch_id: 74, batch_time: 0.00871s, loss: 0.43535

```

预测：

在参数epoch：20，num_factors：8及用指标HR@10、NDCG@10与论文进行对比：

本例：

```
2020-04-28 12:17:56,541-INFO: epoch: 20, epoch_time: 101.68907s, HR: 0.57268, NDCG: 0.32499
```

论文：

```
HR: 0.688, NDCG: 0.410
```

