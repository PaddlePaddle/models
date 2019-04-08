
<div align="center">
  <h3>
    <a href="docs/tutorial.md">
      算法原理介绍
    </a>
    <span> | </span>
    <a href="docs/usage.md">
      使用文档
    </a>
    <span> | </span>
    <a href="docs/demo.md">
      示例文档
    </a>
    <span> | </span>
    <a href="docs/model_zoo.md">
      Model Zoo
    </a>
  </h3>
</div>


---
# PaddleSlim模型压缩工具库

PaddleSlim是PaddlePaddle框架的一个子模块。PaddleSlim首次在PaddlePaddle 1.4版本中发布。在PaddleSlim中，实现了目前主流的网络剪枝、参数量化、模型蒸馏三种压缩策略，主要用于压缩图像领域模型。在后续版本中，会添加更多的压缩策略，以及完善对NLP领域模型的支持。

## 目录
- [特色](#特色)
- [架构介绍](#架构介绍)
- [功能列表](#功能列表)
- [实验结果](#简要实验结果)

## 特色

Paddle-Slim工具库有以下特色：

###  接口简单

- 以配置文件方式集中管理可配参数，方便实验管理
- 在普通模型训练脚本上，添加极少代码即可完成模型压缩

详见：[使用示例](docs/demo.md)

### 效果更好

- 剪切与蒸馏压缩策略效果优于其他类似工具

详见：[效果数据](docs/model_zoo.md)

### 功能更强更灵活

- 剪切压缩过程自动化
- 剪切压缩策略支持更多网络结构
- 蒸馏支持多种方式，用户可自定义组合loss
- 支持快速配置多种压缩策略组合使用

详见：[使用说明](docs/usage.md)

## 架构介绍

这里简要介绍模型压缩工具实现的整体原理，便于理解使用流程。
**图 1**为模型压缩工具的架构图，从上到下为API依赖关系。蒸馏模块、量化模块和剪切模块都间接依赖底层的paddle框架。考虑到这种依赖关系和版本的管理，我们将模型压缩工具作为了paddle框架的一部分，所以已经安装普通版本paddle的用户需要重新下载安装支持模型压缩功能的paddle，才能使用压缩功能。

<p align="center">
<img src="docs/images/framework_0.png" height=452 width=706 hspace='10'/> <br />
<strong>图 1</strong>
</p>

如**图 1**所示，最上层的紫色模块为用户接口，在Python脚本中调用模型压缩功能时，只需要构造一个Compressor对象即可，在[使用文档](docs/usage.md)中会有详细说明。

我们将每个压缩算法称为压缩策略，在迭代训练模型的过程中调用用户注册的压缩策略完成模型压缩，如**图2**所示。其中，模型压缩工具封装好了模型训练逻辑，用户只需要提供训练模型需要的网络结构、数据、优化策略（optimizer）等，在[使用文档](docs/usage.md)会对此详细介绍。

<p align="center">
<img src="docs/images/framework_1.png" height=255 width=646 hspace='10'/> <br />
<strong>图 2</strong>
</p>

## 功能列表


### 剪切

- 支持敏感度和uniform两种方式
- 支持VGG、ResNet、MobileNet等各种类型的网络
- 支持用户自定义剪切范围

### 量化训练

- 支持动态和静态两种量化训练方式
- 支持以float类型模拟int8保存模型
- 支持以int8类型保存模型
- 支持以兼容paddle mobile的格式保存模型

### 蒸馏

- 支持在teacher网络和student网络任意层添加组合loss
  - 支持FSP loss
  - 支持L2 loss
  - 支持softmax with cross-entropy loss

### 其它功能

- 支持配置文件管理压缩任务超参数
- 支持多种压缩策略组合使用
- 蒸馏和剪切压缩过程支持checkpoints功能

## 简要实验结果

### 量化训练

评估实验所使用数据集为ImageNet1000类数据，且以top-1准确率为衡量指标：

| Model | FP32| int8(A:abs_max, W:abs_max) | int8, (A:moving_average_abs_max, W:abs_max) |int8, (A:abs_max, W:channel_wise_abs_max) |
|:---|:---:|:---:|:---:|:---:|
|MobileNetV1|70.916%|71.008%|70.84%|71.00%|
|ResNet50|76.352%|76.612%|76.456%|76.73%|

### 卷积核剪切

数据：ImageNet 1000类
模型：MobileNetV1
原始模型大小：17M
原始精度（top5/top1）: 89.54% / 70.91%

#### Uniform剪切

| FLOPS |model size| 精度损失（top5/top1）|精度（top5/top1） |
|---|---|---|---|
| -50%|-47.0%(9.0M)|-0.41% / -1.08%|89.13% / 69.83%|
| -60%|-55.9%(7.5M)|-1.34% / -2.67%|88.22% / 68.24%|
| -70%|-65.3%(5.9M)|-2.55% / -4.34%|86.99% / 66.57%|

#### 基于敏感度迭代剪切

| FLOPS |model size| 精度损失（top5/top1）|精度（top5/top1） |
|---|---|---|---|
| -50%|-59.4%(6.9M)|-1.39% / -2.71%|88.15% / 68.20%|
| -60%|-70.6%(5.0M)|-2.53% / -4.60%|87.01% / 66.31%|
| -70%|-78.8%(3.6M)|-4.24% / -7.50%|85.30% / 63.41%|

### 蒸馏

数据：ImageNet 1000类
模型：MobileNetV1

|- |精度(top5/top1) |收益(top5/top1)|
|---|---|---|
| 单独训| 89.54% / 70.91%| - |
| ResNet50蒸馏训| 90.92% / 71.97%| +1.28% / +1.06%|

### 蒸馏后量化

数据：ImageNet 1000类
模型：MobileNetV1

|压缩策略 |精度(top5/top1) |模型大小|
|---|---|---|
| 单独训无量化|89.54% / 70.91%|17M|
| ResNet50蒸馏|90.92% / 71.97%|17M|
| ResNet50蒸馏训 + 量化|90.94% / 72.08%|
| 剪切-50% FLOPS + 量化|89.11% / 69.70%|

### 剪切后量化

数据：ImageNet 1000类
模型：MobileNetV1

| 剪切FLOPS |剪切精度（top5/top1） |剪切+量化（dynamic）
|---|---|---|
|-50%|89.13% / 69.83%|89.11% / 69.70%|
