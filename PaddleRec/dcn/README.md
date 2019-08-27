# Deep & Cross Network

以下是本例的简要目录结构及说明：

```text
.
├── README.md            # 文档
├── local_train.py       # 本地训练脚本
├── infer.py             # 预测脚本
├── network.py           # 网络结构
├── config.py            # 参数配置
├── reader.py            # 读取数据相关的函数
├── data/
    ├── download.sh      # 下载数据脚本
    ├── preprocess.py    # 数据预处理脚本

```

## 介绍
DCN模型介绍可以参阅论文[Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)

## 环境
- PaddlePaddle 1.5.1

## 数据下载

我们在Criteo数据集训练测试DCN。整个数据集包含约4500万条记录。每一行第一列是label，表示该条广告是否被点击，剩下的是13个整数型特征(I1 - I13)和26个离散型特征(C1 - C26)。

数据下载命令
```bash
cd data && sh download.sh
```

## 数据处理

- 根据论文，使用前6天的数据进行训练（大约41million），第7天的数据一半做valid一半做test。基本上是将数据集按照9:0.5:0.5切分，需要注意的是train数据是前90%。而如xdeepfm等论文实验中8:1:1，并且是完全打乱的。
- 论文对整数型特征数据使用了log transform，因为只有I2最小值为-3，其余最小值为0，所以对I2采用log(4 + l2_value)对其余采用log(1 + l*_value)。
- 统计每个离散型特征（即C1 - C26）出现的不同feature id，存在大量的低频feature id。所以需要对低频feature id进行过滤，缩小embedding matrix大小。代码默认设置的频率是10，去掉了大量低频feature id。

数据预处理命令
```bash
python preprocess.py
```

数据预处理后，训练数据在train中，验证和测试数据在test_valid中，vocab存储离散型特征过滤低频后的feature id。并统计了整数型特征的最小/最大值，离散型特征的feature id数量。

## 本地训练

```bash
nohup python -u local_train.py > train.log &
```

## 本地预测
```bash
nohup python -u infer.py --test_epoch 2 > test.log &
```
注意：最后一行的auc是整个预测数据集的auc

## 结果
本结果在Linux CPU机器上使用dataset开启20线程训练，batch size为512。经过150000 steps（~1.87 epoch）后，预测实验结果如下：
```text
loss: [0.44703564]      auc_val: [0.80654419]
```
