# LRC 局部Rademachar复杂度正则化
本目录包括了一种基于局部rademacher复杂度的新型正则（LRC）的图像分类模型。该模型将LRC正则和[DARTS](https://arxiv.org/abs/1806.09055)网络相结合，在CIFAR-10数据集中得到了97.3%的准确率。

---
# 内容

- [安装](#安装)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型性能](#模型性能)

## 安装

在当前目录下运行样例代码需要PadddlePaddle Fluid的v.1.2.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/install/index_cn.html#paddlepaddle)中的说明来更新PaddlePaddle。

## 数据准备

第一次使用CIFAR-10数据集时，您可以通过如果命令下载：

    sh ./dataset/download.sh

请确保您的环境有互联网连接。数据会下载到`train.py`同目录下的`dataset/cifar/cifar-10-batches-py`。如果下载失败，您可以自行从https://www.cs.toronto.edu/~kriz/cifar.html上下载cifar-10-python.tar.gz并解压到上述位置。

## 模型训练

数据准备好后，可以通过如下命令开始训练：

    python -u train_mixup.py \
        --batch_size=80 \
        --auxiliary \
        --weight_decay=0.0003 \
        --learning_rate=0.025 \
        --lrc_loss_lambda=0.7 \
        --cutout
- 通过设置 ```export CUDA_VISIBLE_DEVICES=0```指定单张GPU训练。
- 可选参数见：

    python train_mixup.py --help

**数据读取器说明：**

* 数据读取器定义在`reader.py`中
* 输入图像尺寸统一变换为32 * 32
* 训练时将图像填充为40 * 40然后随机剪裁为原输入图像大小
* 训练时图像随机水平翻转
* 对图像每个像素做归一化处理
* 训练时对图像做随机遮挡
* 训练时对输入图像做随机洗牌

**模型配置：**

* 使用辅助损失，辅助损失权重为0.4
* 使用dropout，随机丢弃率为0.2
* 设置lrc\_loss\_lambda为0.7

**训练策略：**

* 采用momentum优化算法训练，momentum=0.9
* 权重衰减系数为0.0001
* 采用正弦学习率衰减，初始学习率为0.025
* 总共训练600轮
* 对卷积权重采用Xaiver初始化，对batch norm权重采用固定初始化，对全连接层权重采用高斯初始化
* 对batch norm和全连接层偏差采用固定初始化，不对卷积设置偏差


## 模型性能
下表为该模型在CIFAR-10数据集上的性能：

| 模型 | 平均top1 | 平均top5 |
| ----- | -------- | -------- |
| [DARTS-LRC](https://paddlemodels.bj.bcebos.com/autodl/fluid_rademacher.tar.gz) | 97.34 | 99.75 |
