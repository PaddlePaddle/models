
# 基于DNN模型的点击率预估模型

## 介绍
本模型实现了下述论文中提出的DNN模型：

```text
@inproceedings{guo2017deepfm,
  title={DeepFM: A Factorization-Machine based Neural Network for CTR Prediction},
  author={Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li and Xiuqiang He},
  booktitle={the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI)},
  pages={1725--1731},
  year={2017}
}
```

## 运行环境
需要先安装PaddlePaddle Fluid，然后运行：

```shell
pip install -r requirements.txt
```

## 数据集
本文使用的是Kaggle公司举办的[展示广告竞赛](https://www.kaggle.com/c/criteo-display-ad-challenge/)中所使用的Criteo数据集。

每一行是一次广告展示的特征，第一列是一个标签，表示这次广告展示是否被点击。总共有39个特征，其中13个特征采用整型值，另外26个特征是类别类特征。测试集中是没有标签的。

下载数据集：
```bash
cd data && ./download.sh && cd ..
```

## 模型
本例子只实现了DeepFM论文中介绍的模型的DNN部分，DeepFM会在其他例子中给出。


## 数据准备
处理原始数据集，整型特征使用min-max归一化方法规范到[0, 1]，类别类特征使用了one-hot编码。原始数据集分割成两部分：90%用于训练，其他10%用于训练过程中的验证。

## 训练
训练的命令行选项可以通过`python train.py -h`列出。

### 单机训练：
```bash
python train.py \
        --train_data_path data/raw/train.txt \
        2>&1 | tee train.log
```

训练到第1轮的第40000个batch后，测试的AUC为0.801178，误差（cost）为0.445196。

### 分布式训练

本地启动一个2 trainer 2 pserver的分布式训练任务，分布式场景下训练数据会按照trainer的id进行切分，保证trainer之间的训练数据不会重叠，提高训练效率

```bash
sh cluster_train.sh
```

## 预测
预测的命令行选项可以通过`python infer.py -h`列出。

对测试集进行预测：
```bash
python infer.py \
        --model_path models/pass-0/ \
        --data_path data/raw/valid.txt
```
注意：infer.py跑完最后输出的AUC才是整个预测文件的整体AUC。

## 在百度云上运行集群训练
1. 参考文档 [在百度云上启动Fluid分布式训练](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/user_guides/howto/training/train_on_baidu_cloud_cn.rst) 在百度云上部署一个CPU集群。
1. 用preprocess.py处理训练数据生成train.txt。
1. 将train.txt切分成集群机器份，放到每台机器上。
1. 用上面的 `分布式训练` 中的命令行启动分布式训练任务.

## 在PaddleCloud上运行集群训练
如果你正在使用PaddleCloud做集群训练，你可以使用```cloud.py```这个文件来帮助你提交任务，```trian.py```中所需要的参数可以通过PaddleCloud的环境变量来提交。