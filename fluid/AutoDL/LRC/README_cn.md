# LRC 局部Rademachar复杂度正则化
为了在深度神经网络中提升泛化能力，正则化的选择十分重要也具有挑战性。本目录包括了一种基于局部rademacher复杂度的新型正则（LRC）的图像分类模型。十分感谢[DARTS](https://arxiv.org/abs/1806.09055)模型对本研究提供的帮助。该模型将LRC正则和DARTS网络相结合，在CIFAR-10数据集中得到了很出色的效果。代码和文章一同发布
> [An Empirical Study on Regularization of Deep Neural Networks by Local Rademacher Complexity](https://arxiv.org/abs/1902.00873)\
> Yingzhen Yang, Xingjian Li, Jun Huan.\
> _arXiv:1902.00873_.

---
# 内容

- [安装](#安装)
- [数据准备](#数据准备)
- [模型训练](#模型训练)

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


## 引用

  - DARTS: Differentiable Architecture Search [`论文`](https://arxiv.org/abs/1806.09055)
  - Differentiable Architecture Search in PyTorch [`代码`](https://github.com/quark0/darts)
