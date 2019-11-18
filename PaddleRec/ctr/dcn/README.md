# Deep & Cross Network

以下是本例的简要目录结构及说明：

```text
.
├── README.md                       # 文档
├── local_train.py                  # 本地训练脚本
├── infer.py                        # 预测脚本
├── network.py                      # 网络结构
├── config.py                       # 参数配置
├── reader.py                       # 读取数据相关的函数
├── utils.py                        # 通用函数
├── data/
    ├── download.sh                 # 下载数据脚本
    ├── preprocess.py               # 数据预处理脚本
├── dist_data/
    ├── dist_data_download.sh       # 下载单机模拟多机小样本数据脚本
    ├── preprocess_dist.py          # 小样本数据预处理脚本

```

## 介绍
DCN模型介绍可以参阅论文[Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)

## 环境
- **目前模型库下模型均要求使用PaddlePaddle 1.6及以上版本或适当的develop版本**

## 数据下载

我们在Criteo数据集训练测试DCN。整个数据集包含约4500万条记录。每一行第一列是label，表示该条广告是否被点击，剩下的是13个整数型特征(I1 - I13)和26个离散型特征(C1 - C26)。

数据下载命令
```bash
cd data && python download.py
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
训练过程中每隔固定的steps（默认为100）输出当前total loss(logloss + 正则), log loss和auc，可以在args.py中调整print_steps。

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

## 多机训练
首先使用命令下载并预处理小规模样例数据集：
```bash
cd dist_data && python dist_download.py && cd ..
```
运行命令本地模拟多机场景，默认使用2 X 2，即2个pserver，2个trainer的方式组网训练。

**注意：在多机训练中，建议使用Paddle 1.6版本以上或[最新版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/Tables.html#whl-dev)。**

```bash
# 该sh不支持Windows
sh cluster_train.sh
```
参数说明：
- train_data_dir: 训练数据目录
- model_output_dir: 模型保存目录
- is_local: 是否单机本地训练(单机模拟多机分布式训练是为0)
- is_sparse: embedding是否使用sparse。如果没有设置，默认是False
- role: 进程角色(pserver或trainer)
- endpoints: 所有pserver地址和端口
- current_endpoint: 当前pserver(role是pserver)端口和地址
- trainers: trainer数量

其他参数见cluster_train.py

预测
```bash
python infer.py --model_output_dir cluster_model --test_epoch 10 --test_valid_data_dir dist_data/dist_test_valid_data --vocab_dir dist_data/vocab --cat_feat_num dist_data/cat_feature_num.txt
```
注意:

- 本地模拟需要关闭代理，e.g. unset http_proxy, unset https_proxy

- 0号trainer保存模型参数

- 每次训练完成后需要手动停止pserver进程，使用以下命令查看pserver进程：

>ps -ef | grep python

- 数据读取使用dataset模式，目前仅支持运行在Linux环境下
