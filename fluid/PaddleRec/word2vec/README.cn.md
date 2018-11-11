
# 基于skip-gram的word2vector模型

## 介绍


## 运行环境
需要先安装PaddlePaddle Fluid

## 数据集
数据集使用的是来自Matt Mahoney(http://mattmahoney.net/dc/textdata.html)的维基百科文章数据集enwiki8.

下载数据集：
```bash
cd data && ./download.sh && cd ..
```

## 模型
本例子实现了一个skip-gram模式的word2vector模型。


## 数据准备
对数据进行预处理以生成一个词典。

```bash
python preprocess.py --data_path data/enwik8 --dict_path data/enwik8_dict
```

## 训练
训练的命令行选项可以通过`python train.py -h`列出。

### 单机训练：

```bash
python train.py \
        --train_data_path data/enwik8 \
        --dict_path data/enwik8_dict \
        2>&1 | tee train.log
```

### 分布式训练

本地启动一个2 trainer 2 pserver的分布式训练任务，分布式场景下训练数据会按照trainer的id进行切分，保证trainer之间的训练数据不会重叠，提高训练效率

```bash
sh cluster_train.sh
```

## 预测


## 在百度云上运行集群训练
1. 参考文档 [在百度云上启动Fluid分布式训练](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/user_guides/howto/training/train_on_baidu_cloud_cn.rst) 在百度云上部署一个CPU集群。
1. 用preprocess.py处理训练数据生成train.txt。
1. 将train.txt切分成集群机器份，放到每台机器上。
1. 用上面的 `分布式训练` 中的命令行启动分布式训练任务.
