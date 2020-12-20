# Fibinet

 以下是本例的简要目录结构及说明： 

```
├── README.md # 文档
├── requirements.txt   # 需要的安装包
├── net.py # Fibinet网络文件
├── feed_generator.py # 数据读取文件
├── args.py # 参数脚本
├── get_data.sh # 生成训练数据脚本
├── train.py # 训练文件
├── infer.py # 预测文件
├── train_gpu.sh # gpu训练shell脚本
├── train_cpu.sh # cpu训练shell脚本
├── infer_gpu.sh # gpu预测shell脚本
├── infer_cpu.sh # cpu预测shell脚本
```

## 简介

[《FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction》]( https://arxiv.org/pdf/1905.09433.pdf)是新浪微博机器学习团队发表在RecSys19上的一篇论文，文章指出当前的许多通过特征组合进行CTR预估的工作主要使用特征向量的内积或哈达玛积来计算交叉特征，这种方法忽略了特征本身的重要程度。提出通过使用Squeeze-Excitation network (SENET) 结构动态学习特征的重要性以及使用一个双线性函数来更好的建模交叉特征。
本项目在paddlepaddle上实现FibiNET的网络结构，并在开源数据集Criteo上验证模型效果，

## 数据下载及预处理

数据地址：[Criteo]( https://fleet.bj.bcebos.com/ctr_data.tar.gz)

（1）将原始训练集按9:1划分为训练集和验证集

（2）数值特征（连续特征）进行归一化处理

## 环境

 PaddlePaddle 1.7.0 

 python3.7 

## 单机训练

GPU环境

在train_gpu.sh脚本文件中设置好数据路径、参数。

```sh
CUDA_VISIBLE_DEVICES=0 python train.py --use_gpu 1 \ #使用gpu
                                    --train_files_path ./train_data_full \ #全量训练数据
                                    --model_dir ./model_dir \ #模型路径
                                    --learning_rate 0.001 \ 
                                    --batch_size 1000 \
                                    --epochs 10 \
                                    --reduction_ratio 3 \ #SENET超参数
                                    --dropout_rate 0.5 
                                    --embedding_size 10
```

修改脚本的可执行权限并运行

```
./train_gpu.sh
```

CPU环境

在train_cpu.sh脚本文件中设置好数据路径、参数。

```sh
python train.py --use_gpu 0 \ #使用cpu
                --train_files_path ./train_data_full \ #全量训练数据
                --model_dir ./model_dir \ #模型路径
                --learning_rate 0.001 \ 
                --batch_size 1000 \
                --epochs 10 \
                --reduction_ratio 3 \ #SENET超参数
                --dropout_rate 0.5 
                --embedding_size 10
```

修改脚本的可执行权限并运行

```
./train_cpu.sh
```

## 单机预测

GPU环境

在infer_gpu.sh脚本文件中设置好数据路径、参数。

```sh
CUDA_VISIBLE_DEVICES=0 python train.py --use_gpu 0 \ #使用gpu
                                    --test_files_path ./test_data_full \ #使用全量测试数据
                                    --model_dir ./model_dir \ #模型路径
                                    --test_epoch 10 #选择哪个epoch的模型参数进行预测
```

修改脚本的可执行权限并运行

```
./infer_gpu.sh
```

CPU环境

在infer_cpu.sh脚本文件中设置好数据路径、参数。

```sh
python train.py --use_gpu 0 \ #使用cpu
                --test_files_path ./test_data_full \ #使用全量测试数据
                --model_dir ./model_dir \ #模型路径
                --test_epoch 10 #选择哪个epoch的模型参数进行预测
```

修改脚本的可执行权限并运行

```
./infer_cpu.sh
```

## 模型效果

训练：

```
2020-06-10 23:34:45,195-INFO: epoch_id: 0, batch_id: 33952, batch_time: 1.26086s, loss: 0.44914, auc: 0.79089
2020-06-10 23:34:46,369-INFO: epoch_id: 0, batch_id: 33953, batch_time: 1.17280s, loss: 0.46410, auc: 0.79089
2020-06-10 23:34:47,413-INFO: epoch_id: 0, batch_id: 33954, batch_time: 1.04139s, loss: 0.43496, auc: 0.79089
2020-06-10 23:34:48,248-INFO: epoch_id: 0, batch_id: 33955, batch_time: 0.83510s, loss: 0.45980, auc: 0.79089
2020-06-10 23:34:49,379-INFO: epoch_id: 0, batch_id: 33956, batch_time: 1.13043s, loss: 0.46738, auc: 0.79089
2020-06-10 23:34:50,392-INFO: epoch_id: 0, batch_id: 33957, batch_time: 1.01046s, loss: 0.46724, auc: 0.79089
2020-06-10 23:34:51,440-INFO: epoch_id: 0, batch_id: 33958, batch_time: 1.04752s, loss: 0.44079, auc: 0.79089
```

