# DeepFM动态图

以下是本例的简要目录结构及说明：

```text
.
├── README.md                       # 文档
├── train.py                        # 本地训练脚本
├── infer.py                        # 本地预测脚本
├── network.py                      # 网络结构
├── data_reader.py                  # 读取数据相关的函数
├── utility.py                      # 参数设置和通用函数
├── data/
    ├── download_preprocess.py      # 下载并预处理数据脚本
    ├── preprocess.py               # 数据预处理脚本

```

## 介绍
本模型使用PaddlePaddle **动态图** 复现了DeepFM模型。

DeepFM模型介绍可以参阅论文[DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)

## 环境
- **目前本模型要求使用PaddlePaddle 1.7或最新develop版本**

## 数据下载和预处理

我们在[Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge/)数据集训练测试DeepFM。整个数据集包含约4500万条记录。每一行第一列是label，表示该条广告是否被点击，剩下的是13个整数型特征(I1 - I13)和26个离散型特征(C1 - C26)。

通过min-max normalize将连续特征转换到 [0, 1]区间，并去除离散型特征中出现少于10次的特征。整个数据集被划分为两部分：90%用来训练，10%用来评估模型效果。

下载并预处理数据命令:
```bash
cd data && python download_preprocess.py && cd ..
```

执行完命令后将生成三个文件夹: train_data, test_data和aid_data。

train_data包含90%数据，test_data包含剩下的10%数据，aid_data中有一个生成或下载（节约用户生成特征字典时间）的特征字典feat_dict_10.pkl2。

## 训练模型

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py > train.log 2>&1 &
```

每一轮数据训练结束后会测试模型效果。

加载已经存在的模型并继续训练:

```bash
# 加载保存的epoch_0并继续训练
CUDA_VISIBLE_DEVICES=0 python -u train.py --checkpoint=models/epoch_0 > train.log 2>&1 &
```

## 预测模型

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py --checkpoint=models/epoch_0
```

加载models/epoch_0的模型，对test_data中数据进行预测，评估模型效果。注意：最后一行才是整个test数据集的auc。

## 效果
```text
test auc of epoch 0 is 0.78+
```

第一轮数据训练结束后，test auc为0.78+。

继续训练模型易出现过拟合现象，可以通过评估模型选择效果最好的模型作为最终训练结果。
