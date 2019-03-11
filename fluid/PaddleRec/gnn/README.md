# SR-GNN

以下是本例的简要目录结构及说明：

```text
.
├── README.md            # 文档
├── train.py             # 训练脚本
├── infer.py             # 预测脚本
├── network.py           # 网络结构
├── cluster_train.py     # 多机训练
├── cluster_train.sh     # 多机训练脚本
├── reader.py            # 和读取数据相关的函数
├── data/
    ├── download.sh         # 下载数据的脚本
    ├── preprocess.py       # 数据预处理

```

## 简介

SR-GNN模型的介绍可以参阅论文[Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)。

本文解决的是Session-based Recommendation这一问题,过程大致分为以下四步：

是对所有的session序列通过有向图进行建模。

然后通过GNN，学习每个node（item）的隐向量表示

然后通过一个attention机制得到每个session的embedding

最后通过一个softmax层进行全表预测

我们复现了论文效果，在DIGINETICA数据集上P@20可以达到50.7


## 数据下载及预处理

使用DIGINETICA数据集，数据来自：http://cikm2016.cs.iupui.edu/cikm-cup。可以按照下述过程操作获得数据集以及进行简单的数据预处理。

* Step 1: 运行如下命令,下载DIGINETICA数据集并进行预处理
```
cd data && sh download.sh
```

* Step 2: 产生训练集、测试集和config文件
```
python preprocess.py
cd ..
```
运行之后在data文件夹下会产生diginetica文件夹，里面包含config.txt、test.txt  train.txt三个文件


## 训练

可以参考下面不同场景下的运行命令就行训练，还可以指定诸如batch_size，lr等参数，具体的配置说明可通过运行下列代码查看
```
python train.py -h
```

gpu 单机单卡训练
``` bash
CUDA_VISIBLE_DEVICES=1 python -u train.py --use_cuda 1 > log.txt 2>&1 &
```

cpu 单机训练
``` bash
python -u train.py --use_cuda 0 > log.txt 2>&1 &
```

值得注意的是上述单卡训练可以通过加--parallel 1参数使用Parallel Executor来进行加速


## 训练结果示例

我们在Tesla K40m单GPU卡上训练的日志如下所示(以实际输出为准)
```text
W0308 16:08:24.249840  1785 device_context.cc:263] Please NOTE: device: 0, CUDA Capability: 35, Driver API Version: 9.0, Runtime API Version: 8.0
W0308 16:08:24.249974  1785 device_context.cc:271] device: 0, cuDNN Version: 7.0.
2019-03-08 16:08:38,079 - INFO - load data complete
2019-03-08 16:08:38,080 - INFO - begin train
2019-03-08 16:09:07,605 - INFO - step: 500, loss: 10.2052, train_acc: 0.0088
2019-03-08 16:09:36,940 - INFO - step: 1000, loss: 9.7192, train_acc: 0.0320
2019-03-08 16:10:08,617 - INFO - step: 1500, loss: 8.9290, train_acc: 0.1350
...
2019-03-08 16:16:01,151 - INFO - model saved in ./saved_model/epoch_0
...
```

## 预测
运行如下命令即可开始预测。可以通过参数指定开始和结束的epoch轮次。

```
CUDA_VISIBLE_DEVICES=3 python infer.py
```

## 预测结果示例
```text
W0308 16:41:56.847339 31709 device_context.cc:263] Please NOTE: device: 0, CUDA Capability: 35, Driver API Version: 9.0, Runtime API Version: 8.0
W0308 16:41:56.847705 31709 device_context.cc:271] device: 0, cuDNN Version: 7.0.
2019-03-08 16:42:20,420 - INFO - TEST --> loss: 5.8865, Recall@20: 0.4525
2019-03-08 16:42:45,153 - INFO - TEST --> loss: 5.5314, Recall@20: 0.5010
2019-03-08 16:43:10,233 - INFO - TEST --> loss: 5.5128, Recall@20: 0.5047
...
```
