# SR-GNN

以下是本例的简要目录结构及说明：

```text
.
├── README.md            # 文档
├── train.py             # 训练脚本
├── infer.py             # 预测脚本
├── network.py           # 网络结构
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

然后通过一个attention架构模型得到每个session的embedding

最后通过一个softmax层进行全表预测

我们复现了论文效果，在DIGINETICA数据集上P@20可以达到50.7


## 数据下载及预处理

使用[DIGINETICA](http://cikm2016.cs.iupui.edu/cikm-cup)数据集。可以按照下述过程操作获得数据集以及进行简单的数据预处理。

* Step 1: 运行如下命令,下载DIGINETICA数据集并进行预处理
```
cd data && sh download.sh
```

* Step 2: 产生训练集、测试集和config文件
```
python preprocess.py --dataset diginetica
cd ..
```
运行之后在data文件夹下会产生diginetica文件夹，里面包含config.txt、test.txt  train.txt三个文件

生成的数据格式为:(session_list,
label_list)。

其中session_list是一个session的列表，其中每个元素都是一个list，代表不同的session。label_list是一个列表，每个位置的元素是session_list中对应session的label。

例子：session_list=[[1,2,3], [4], [7,9]]。代表这个session_list包含3个session，第一个session包含的item序列是1,2,3，第二个session只有1个item 4，第三个session包含的item序列是7，9。

label_list = [6, 9,
1]。代表[1,2,3]这个session的预测label值应该为6，后两个以此类推。

提示：

* 如果您想使用自己业务场景下的数据，只要令数据满足上述格式要求即可
* 本例中的train.txt和test.txt两个文件均为二进制文件


## 训练

可以参考下面不同场景下的运行命令进行训练，还可以指定诸如batch_size，lr(learning rate)等参数，具体的配置说明可通过运行下列代码查看
```
python train.py -h
```

gpu 单机单卡训练
``` bash
CUDA_VISIBLE_DEVICES=1 python -u train.py --use_cuda 1 > log.txt 2>&1 &
```

cpu 单机训练
``` bash
CPU_NUM=1 python -u train.py --use_cuda 0 > log.txt 2>&1 &
```

值得注意的是上述单卡训练可以通过加--use_parallel 1参数使用Parallel Executor来进行加速。


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
