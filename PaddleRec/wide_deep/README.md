# wide&deep

 以下是本例的简要目录结构及说明： 



```
├── README.md # 文档
├── requirements.txt   # 需要的安装包
├── net.py # wide&deep网络文件
├── utils.py # 通用函数
├── args.py # 参数脚本
├── create_data.sh # 生成训练数据脚本
├── data_preparation.py # 数据预处理脚本
├── train.py # 训练文件
├── infer.py # 预测文件
├── train_gpu.sh # gpu训练shell脚本
├── train_cpu.sh # cpu训练shell脚本
├── infer_gpu.sh # gpu预测shell脚本
├── infer_cpu.sh # cpu预测shell脚本
```

## 简介

[《Wide & Deep Learning for Recommender Systems》]( https://arxiv.org/pdf/1606.07792.pdf)是Google 2016年发布的推荐框架，wide&deep设计了一种融合浅层（wide）模型和深层（deep）模型进行联合训练的框架，综合利用浅层模型的记忆能力和深层模型的泛化能力，实现单模型对推荐系统准确性和扩展性的兼顾。从推荐效果和服务性能两方面进行评价：

1. 效果上，在Google Play 进行线上A/B实验，wide&deep模型相比高度优化的Wide浅层模型，app下载率+3.9%。相比deep模型也有一定提升。
2. 性能上，通过切分一次请求需要处理的app 的Batch size为更小的size，并利用多线程并行请求达到提高处理效率的目的。单次响应耗时从31ms下降到14ms。

本例在paddlepaddle上实现wide&deep并在开源数据集Census-income Data上验证模型效果，在测试集上的平均acc和auc分别为：

> mean_acc: 0.76195
>
> mean_auc: 0.90335

## 数据下载及预处理

数据地址： 

[adult.data](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)

[adult.test](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test)

在create_data.sh脚本文件中添加文件的路径，并运行脚本。

```sh
mkdir train_data
mkdir test_data
mkdir data
train_path="data/adult.data" #原始训练数据
test_path="data/adult.test" #原始测试数据
train_data_path="train_data/train_data.csv" #处理后的训练数据
test_data_path="test_data/train_data.csv" #处理后的测试数据

pip install -r requirements.txt #安装必需包

wget -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

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
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 40 \ #训练轮次
                                        --batch_size 40 \ #batch大小
                                        --use_gpu 1 \ #使用gpu训练
                                        --train_data_path 'train_data/train_data.csv' \ #训练数据
                                        --model_dir 'model_dir' #模型保存路径
                                        --hidden1_units 75 \ #deep网络隐层大小
                                        --hidden2_units 50 \
                                        --hidden3_units 25
                
```

修改脚本的可执行权限并运行

```sh
./train_gpu.sh
```

CPU环境

在train_cpu.sh脚本文件中设置好数据路径、参数。

```sh
python train.py --epochs 40 \ #训练轮次
                --batch_size 40 \ #batch大小
                --use_gpu 0 \ #使用cpu训练
                --train_data_path 'train_data/train_data.csv' \ #训练数据
                --model_dir 'model_dir' #模型保存路径
                --hidden1_units 75 \ #deep网络隐层大小
                --hidden2_units 50 \
                --hidden3_units 25
                
```

修改脚本的可执行权限并运行

```
./train_cpu.sh
```

## 单机预测

GPU环境

在infer_gpu.sh脚本文件中设置好数据路径、参数。

```sh
python infer.py --batch_size 40 \ #batch大小
                --use_gpu 0 \ #使用cpu训练
                --test_epoch 39 \ #选择那一轮的模型用来预测
                --test_data_path 'test_data/test_data.csv' \ #测试数据
                --model_dir 'model_dir' \ #模型路径
                --hidden1_units 75 \ #隐层单元个数
                --hidden2_units 50 \
                --hidden3_units 25
                
```

修改脚本的可执行权限并运行

```sh
./infer_gpu.sh
```

CPU环境

在infer_cpu.sh脚本文件中设置好数据路径、参数。

```sh
python infer.py --batch_size 40 \ #batch大小
                --use_gpu 0 \ #使用cpu训练
                --test_epoch 39 \ #选择那一轮的模型用来预测
                --test_data_path 'test_data/test_data.csv' \ #测试数据
                --model_dir 'model_dir' \ #模型路径
                --hidden1_units 75 \ #隐层单元个数
                --hidden2_units 50 \
                --hidden3_units 25
                
```

修改脚本的可执行权限并运行

```
./infer_cpu.sh
```

## 模型效果

在测试集的效果如下：

```
W0422 10:45:12.497740  1218 device_context.cc:237] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 9.2, Runtime API Version: 9.0
W0422 10:45:12.501889  1218 device_context.cc:245] device: 0, cuDNN Version: 7.3.
2020-04-22 10:45:13,804-INFO: batch_id: 0,batch_time: 0.00625s,acc: 0.72500,auc: 0.92790
2020-04-22 10:45:13,809-INFO: batch_id: 1,batch_time: 0.00468s,acc: 0.80000,auc: 0.92321
2020-04-22 10:45:13,814-INFO: batch_id: 2,batch_time: 0.00442s,acc: 0.82500,auc: 0.93003
2020-04-22 10:45:13,819-INFO: batch_id: 3,batch_time: 0.00434s,acc: 0.75000,auc: 0.94108
2020-04-22 10:45:13,824-INFO: batch_id: 4,batch_time: 0.00438s,acc: 0.67500,auc: 0.93013
2020-04-22 10:45:13,829-INFO: batch_id: 5,batch_time: 0.00438s,acc: 0.80000,auc: 0.92201
......
2020-04-22 10:45:15,914-INFO: batch_id: 403,batch_time: 0.00487s,acc: 0.80000,auc: 0.90454
2020-04-22 10:45:15,920-INFO: batch_id: 404,batch_time: 0.00505s,acc: 0.72500,auc: 0.90427
2020-04-22 10:45:15,925-INFO: batch_id: 405,batch_time: 0.00460s,acc: 0.77500,auc: 0.90405
2020-04-22 10:45:15,931-INFO: batch_id: 406,batch_time: 0.00517s,acc: 0.77500,auc: 0.90412
2020-04-22 10:45:15,936-INFO: batch_id: 407,batch_time: 0.00457s,acc: 0.00000,auc: 0.90415
2020-04-22 10:45:15,936-INFO: mean_acc:0.76195,mean_auc:0.90335
```

