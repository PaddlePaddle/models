# ESMM

以下是本例的简要目录结构及说明： 

```
├── README.md			 # 文档
├── net.py				 # ESMM网络结构
├── train.py			 # ESMM模型训练脚本
├── infer.py			 # ESMM模型预测脚本
├── utils.py			 # 通用函数
├── args.py				 # 参数脚本
├── get_data.sh			 # 生成训练数据脚本
├── dataset_generator.py # dataset生成脚本
├── gpu_train.sh		 # gpu训练shell脚本
├── cpu_train.sh		 # cpu训练shell脚本
├── gpu_infer.sh		 # gpu预测shell脚本
├── cpu_infer.sh		 # cpu预测shell脚本
├── vocab_size.txt       #词汇表文件
```

## 简介

不同于CTR预估问题，CVR预估面临两个关键问题：

1. **Sample Selection Bias (SSB)** 转化是在点击之后才“有可能”发生的动作，传统CVR模型通常以点击数据为训练集，其中点击未转化为负例，点击并转化为正例。但是训练好的模型实际使用时，则是对整个空间的样本进行预估，而非只对点击样本进行预估。即是说，训练数据与实际要预测的数据来自不同分布，这个偏差对模型的泛化能力构成了很大挑战。
2. **Data Sparsity (DS)** 作为CVR训练数据的点击样本远小于CTR预估训练使用的曝光样本。

ESMM是发表在 SIGIR’2018 的论文[《Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate》](  https://arxiv.org/abs/1804.07931  )文章基于 Multi-Task Learning 的思路，提出一种新的CVR预估模型——ESMM，有效解决了真实场景中CVR预估面临的数据稀疏以及样本选择偏差这两个关键问题

本项目再Paddlepaddle定义ESMM的网络结构，并在论文的公开数据集[Ali-CCP：Alibaba Click and Conversion Prediction](  https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408  )验证模型的效果（目前只抽取部分数据验证模型的正确性）。

## 环境

 PaddlePaddle 1.7.0 

 python3.7 

## 数据下载及预处理

执行get_data.sh即可获得处理后的数据

```shell
./get_data.sh
```

## 单机训练

GPU环境

在gpu_train.sh脚本文件中设置好数据路径、参数。

```shell
CUDA_VISIBLE_DEVICES=0 python train.py	--use_gpu 1\  #是否使用gpu
                                        --epochs 100\  #训练轮次
                                        --batch_size 64\  #batch_size大小
                                        --embed_size 12\  #每个featsigns的embedding维度
                                        --cpu_num 2\  #cpu数量
                                        --model_dir ./model_dir \  #模型保存路径
                                        --train_data_path ./train_data \  #训练数据路径
                                        --vocab_path ./vocab_size.txt #embedding词汇表大小路径
```

修改脚本的可执行权限并运行

```shell
./gpu_train.sh
```

CPU环境

在cpu_train.sh脚本文件中设置好数据路径、参数。

```shell
python train.py --use_gpu 0\  #是否使用gpu
                --epochs 100\  #训练轮次
                --batch_size 64\  #batch_size大小
                --embed_size 12\  #每个featsigns的embedding维度
                --cpu_num 2\  #cpu数量
                --model_dir ./model_dir \  #模型保存路径
                --train_data_path ./train_data \  #训练数据路径
                --vocab_path ./vocab_size.txt #embedding词汇表大小路径
```

修改脚本的可执行权限并运行

```
./cpu_train.sh
```

## 预测

GPU环境

在gpu_infer.sh脚本文件中设置好数据路径、参数。

```sh
python infer.py --use_gpu 1\  #是否使用gpu
                --batch_size 64\  #batch_size大小
                --test_data_path ./test_data \  #训练数据路径
                --vocab_path ./vocab_size.txt #embedding词汇表大小路径
```

修改脚本的可执行权限并运行

```shell
./gpu_infer.sh
```

CPU环境

在cpu_infer.sh脚本文件中设置好数据路径、参数。

```shell
python infer.py --use_gpu 0\  #是否使用gpu
                --batch_size 64\  #batch_size大小
                --cpu_num 2\  #cpu数量
                --test_data_path ./test_data \  #训练数据路径
                --vocab_path ./vocab_size.txt #embedding词汇表大小路径
```

修改脚本的可执行权限并运行

```
./cpu_infer.sh
```



## 模型效果

目前只抽取部分数据验证模型正确性。模型预测结果实例如下：

> auc_ctr	auc_0.tmp_0		lod: {}
> 	dim: 1
> 	layout: NCHW
> 	dtype: double
> 	data: [0.971812]
> 	
> auc_ctcvr	auc_1.tmp_0		lod: {}
> 	dim: 1
> 	layout: NCHW
> 	dtype: double
> 	data: [0.499668]

