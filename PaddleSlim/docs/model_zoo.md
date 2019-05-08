<div align="center">
  <h3>
    <a href="usage.md">
      使用文档
    </a>
    <span> | </span>
    <a href="demo.md">
      示例文档
    </a>
    <span> | </span>
    <a href="tutorial.md">
      算法原理介绍
    </a>
  </h3>
</div>


---
# Paddle模型压缩工具实验与模型库

## 目录

- [量化实验](#int8量化训练)
- [剪切实验](#剪切实验)
- [蒸馏实验](蒸馏实验)
- [组合实验](组合实验)


## 1. int8量化训练

评估实验所使用数据集为ImageNet 1000类数据, 量化训练前后模型top-5/top-1准确率对比如下：

| Model | FP32| int8(X:abs_max, W:abs_max) | int8, (X:moving_average_abs_max, W:abs_max) |int8, (X:abs_max, W:channel_wise_abs_max) |
|:---|:---:|:---:|:---:|:---:|
|MobileNetV1|[89.54% / 70.91%]()|[89.64% / 71.01%]()|[89.58% / 70.86%]()|[89.75% / 71.13%]()|
|ResNet50|[92.80% / 76.35%]()|[93.12% / 76.77%]()|[93.07% / 76.65%]()|[93.15% / 76.80%]()|

点击表中超链接即可下载预训练模型。

量化训练前后，模型大小的变化对比如下：

| Model       | FP32  | int8(A:abs_max, W:abs_max) | int8, (A:moving_average_abs_max, W:abs_max) | int8, (A:abs_max, W:channel_wise_abs_max) |
| :---        | :---: | :---:                      | :---:                                       | :---:                                     |
| MobileNetV1 | 17M   | 4.8M(-71.76%)               | 4.9M(-71.18%)                                | 4.9M(-71.18%)                              |
| ResNet50    | 99M   | 26M(-73.74%)                | 27M(-72.73%)                                 | 27M(-72.73%)                               |

注：abs_max为动态量化，moving_average_abs_max为静态量化, channel_wise_abs_max是对卷积权重进行分channel量化。


## 2. 剪切实验

数据： ImageNet 1000类
模型：MobileNetV1
原始模型大小：17M
原始精度（top5/top1）: 89.54% / 70.91%

### 2.1 基于敏感度迭代剪切

#### 实验说明

分步剪切，每步剪掉模型7%的FLOPS.
optimizer配置如下：

```
epoch_size=5000
boundaries = [30, 60, 90, 120] * epoch_size # for -50% FLOPS
#boundaries = [35, 65, 95, 125] * epoch_size # for -60% FLOPS
#boundaries = [50, 80, 110, 140] * epoch_size # for -70% FLOPS
values = [0.01, 0.1, 0.01, 0.001, 0.0001]
optimizer = fluid.optimizer.Momentum(
        momentum=0.9,
        learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
        regularization=fluid.regularizer.L2Decay(1e-4))
```

#### 实验结果


| FLOPS |model size| 精度（top5/top1） |下载模型|
|---|---|---|---|
| -50%|-59.4%(6.9M) |88.22% / 68.41%   |[点击下载](https://paddle-slim-models.bj.bcebos.com/sensitive_filter_pruning_0.5_model.tar.gz)|
| -60%|-70.6%(5.0M)|87.01% / 66.31% |[点击下载](https://paddle-slim-models.bj.bcebos.com/sensitive_filter_pruning_0.6_model.tar.gz)|
| -70%|-78.8%(3.6M)|85.30% / 63.41%  |[点击下载](https://paddle-slim-models.bj.bcebos.com/sensitive_filter_pruning_0.7_model.tar.gz)|

### 2.2 基于敏感度一次性剪切

#### 实验说明

一步剪切掉50%FLOPS, 然后fine-tune 120个epoch.

optimizer配置如下：

```
epoch_size=5000
boundaries = [30, 60, 90] * epoch_size
values = [0.1, 0.01, 0.001, 0.0001]
optimizer = fluid.optimizer.Momentum(
        momentum=0.9,
        learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
        regularization=fluid.regularizer.L2Decay(1e-4))
```

#### 实验结果

| FLOPS |model size|精度（top5/top1） |模型下载|
|---|---|---|---|
| -50%|-61.2%(6.6M)|  88.47% / 68.68% |[点击下载](https://paddle-slim-models.bj.bcebos.com/sensitive_filter_pruning_0.5-1step.tar.gz)|

### 2.3 基于敏感度分步剪切

#### 实验说明

1. 一次剪掉20%FLOPS, fine-tune 120个epoch
2. 在上一步基础上，一次剪掉20%FLOPS, fine-tune 120个epoch
3. 在上一步基础上，一次剪掉20%FLOPS, fine-tune 120个epoch

optimizer配置如下：

```
epoch_size=5000
boundaries = [30, 60, 90] * epoch_size
values = [0.1, 0.01, 0.001, 0.0001]
optimizer = fluid.optimizer.Momentum(
        momentum=0.9,
        learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
        regularization=fluid.regularizer.L2Decay(1e-4))
```

#### 实验结果

| FLOPS |精度（top5/top1）|模型下载 |
|---|---|---|
| -20%|90.08% / 71.48% |[点击下载](https://paddle-slim-models.bj.bcebos.com/sensitive_filter_pruning_3step_0.2_model.tar.gz)|
| -36%|89.62% / 70.83%|[点击下载](https://paddle-slim-models.bj.bcebos.com/sensitive_filter_pruning_3step_0.36_model.tar.gz)|
| -50%|88.77% / 69.31%|[点击下载](https://paddle-slim-models.bj.bcebos.com/sensitive_filter_pruning_3step_0.5_model.tar.gz)|


### 2.4 Uniform剪切

#### 实验说明

一次剪掉指定比例的FLOPS，然后fine-tune 120个epoch.

optimizer配置如下：

```
epoch_size=5000
boundaries = [30, 60, 90] * epoch_size
values = [0.1, 0.01, 0.001, 0.0001]
optimizer = fluid.optimizer.Momentum(
        momentum=0.9,
        learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
        regularization=fluid.regularizer.L2Decay(1e-4))
```

#### 实验结果

| FLOPS |model size|精度（top5/top1） |模型下载 |
|---|---|---|---|
| -50%|-47.0%(9.0M) | 89.13% / 69.83%|[点击下载](https://paddle-slim-models.bj.bcebos.com/uniform_filter_pruning_0.5_model.tar.gz)|
| -60%|-55.9%(7.5M)|88.22% / 68.24%| [点击下载](https://paddle-slim-models.bj.bcebos.com/uniform_filter_pruning_0.6_model.tar.gz)|
| -70%|-65.3%(5.9M)|86.99% / 66.57%| [点击下载](https://paddle-slim-models.bj.bcebos.com/uniform_filter_pruning_0.7_model.tar.gz)|


## 3. 蒸馏

数据： ImageNet 1000类
模型：MobileNetV1
原始模型大小：17M
原始精度（top5/top1）: 89.54% / 70.91%

#### 实验说明

用训练好的ResNet50蒸馏训练MobileNetV1, 训练120个epoch. 对第一个block加FSP loss; 对softmax layer的输入加L2-loss.

optimizer配置如下：

```
epoch_size=5000
boundaries = [30, 60, 90] * epoch_size
values = [0.1, 0.01, 0.001, 0.0001]
optimizer = fluid.optimizer.Momentum(
        momentum=0.9,
        learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
        regularization=fluid.regularizer.L2Decay(1e-4))
```

#### 实验结果

|- |精度(top5/top1) |收益(top5/top1)|模型下载 |
|---|---|---|---|
| ResNet50蒸馏训| 90.92% / 71.97%| +1.28% / +1.06%| [点击下载](https://paddle-slim-models.bj.bcebos.com/mobilenetv1_resnet50_distillation_model.tar.gz)|


## 4. 组合实验

### 4.1 蒸馏后量化

#### 实验说明

#### 实验结果

|- |精度(top1) |模型下载 |
|---|---|---|
| ResNet50蒸馏训练+量化| 72.01%| [点击下载]()|


### 4.2 剪切后量化


#### 实验说明

#### 实验结果

| 剪切FLOPS |剪切+量化（dynamic）|模型下载 |
|---|---|---|
| -50%| 69.20%| [点击下载]()|
