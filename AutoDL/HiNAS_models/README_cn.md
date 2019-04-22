# Image Classification Models
本目录下包含6个图像分类模型，都是百度大数据实验室 Hierarchical Neural Architecture Search (HiNAS) 项目通过机器自动发现的模型，在CIFAR-10数据集上达到96.1%的准确率。这6个模型分为两类，前3个没有skip link，分别命名为 HiNAS 0-2号，后三个网络带有skip link，功能类似于Resnet中的shortcut connection，分别命名 HiNAS 3-5号。

---
## Table of Contents
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training a model](#training-a-model)
- [Model performances](#model-performances)

## Installation
最低环境要求:

- PadddlePaddle Fluid >= v0.15.0
- Cudnn >=6.0

如果您的运行环境无法满足要求，可以参考此文档升级PaddlePaddle：[installation document](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)

## Data preparation

第一次训练模型的时候，Trainer会自动下载CIFAR-10数据集，请确保您的环境有互联网连接。

数据集会被下载到Trainer同目录下的`dataset/cifar/cifar-10-python.tar.gz`，如果自动下载失败，您可以自行从 https://www.cs.toronto.edu/~kriz/cifar.html 下载cifar-10-python.tar.gz，然后放到上述位置。


## Training a model
准备好环境后，可以训练模型，训练有2个入口，`train_hinas.py`和`train_hinas_res.py`，前者用来训练0-2号不含skip link的模型，后者用来训练3-5号包含skip link的模型。

训练0~2号不含skip link的模型：
```
python train_hinas.py --model=m_id       # m_id can be 0, 1 or 2.
```
训练3~5号包含skip link的模型：
```
python train_hinas_res.py --model=m_id    # m_id can be 0, 1 or 2.
```

此外，`train_hinas.py`和`train_hinas_res.py` 都支持以下参数：

初始化部分：

- random_flip_left_right：图片随机水平翻转（Default：True）
- random_flip_up_down：图片随机垂直翻转（Default：False）
- cutout：图片随机遮挡（Default：True）
- standardize_image：对图片每个像素做 standardize（Default：True）
- pad_and_cut_image：图片随机padding，并裁剪回原大小（Default：True）
- shuffle_image：训练时对输入图片的顺序做shuffle（Default：True）
- lr_max：训练开始时的learning rate（Default：0.1）
- lr_min：训练结束时的learning rate（Default：0.0001）
- batch_size：训练的batch size（Default：128）
- num_epochs：训练总的epoch（Default：200）
- weight_decay：训练时L2 Regularization大小（Default：0.0004）
- momentum：momentum优化器中的momentum系数（Default：0.9）
- dropout_rate：dropout层的dropout_rate（Default：0.5）
- bn_decay：batch norm层的decay/momentum系数（即moving average decay）大小（Default：0.9）



## Model performances
6个模型使用相同的参数训练：

- learning rate: 0.1 -> 0.0001 with cosine annealing
- total epoch: 200
- batch size: 128
- L2 decay: 0.000400
- optimizer: momentum optimizer with m=0.9 and use nesterov
- preprocess: random horizontal flip + image standardization + cutout

以下是6个模型在CIFAR-10数据集上的准确率：

| model    | round 1 | round 2 | round 3 | max    | avg    |
|----------|---------|---------|---------|--------|--------|
| HiNAS-0  | 0.9548  | 0.9520  | 0.9513  | 0.9548 | 0.9527 |
| HiNAS-1  | 0.9452  | 0.9462  | 0.9420  | 0.9462 | 0.9445 |
| HiNAS-2  | 0.9508  | 0.9506  | 0.9483  | 0.9508 | 0.9499 |
| HiNAS-3  | 0.9607  | 0.9623  | 0.9601  | 0.9623 | 0.9611 |
| HiNAS-4  | 0.9611  | 0.9584  | 0.9586  | 0.9611 | 0.9594 |
| HiNAS-5  | 0.9578  | 0.9588  | 0.9594  | 0.9594 | 0.9586 |
