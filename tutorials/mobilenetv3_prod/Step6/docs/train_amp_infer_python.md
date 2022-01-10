# MobileNetV3

## 目录


- [1. 简介](#1)
- [2. 混合精度训练](#2)
    - [2.1 检查环境](#2.1)
    - [2.2 混合精度O1模式训练](#2.2)
    - [2.3 混合精度O2模式训练](#2.3)
- [3. FAQ](#3)


<a name="1"></a>

## 1. 简介

Paddle 混合精度训练（Auto Mixed Precision, AMP）是指在训练过程中同时使用单精度（FP32）和半精度（FP16），基于NVIDIA GPU提供的Tensor Cores技术，混合精度训练使用FP16和FP32即可达到与使用纯FP32训练相同的准确率，并可加速模型的训练速度。

本文档主要基于Paddle的MobileNetV3模型混合精度训练。

更多关于Paddle AMP的介绍，可以参考[Paddle AMP官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/amp_cn.html)。


<a name="2"></a>

## 2. 混合精度训练

<a name="2.1"></a>

### 2.1 检查环境

混合精度训练的加速依赖于NVIDIA显卡的Tensor Core，目前Tensor Core仅支持Compute Capability在7.0及其以上的显卡，因此在开发混合精度训练之前，首先检查显卡是否支持混合精度训练，检查方法如下：

- 进入python环境，执行如下命令：

```
>>> import paddle
>>> paddle.device.cuda.get_device_capability()
```

以Tesla V100显卡为例，执行上述命令后将打印出如下形式的结果：

```
(7, 0)
```

结果显示该显卡Compute Capability为7.0，因此符合混合精度训练所要求的环境。

此外，若不预先执行上述检查，混合精度训练依旧可以执行，但无法达到性能提升的效果，且在代码执行混合精度训练前会打印UserWarning，以Tesla K40显卡为例：

```
UserWarning: AMP only support NVIDIA GPU with Compute Capability 7.0 or higher, current GPU is: Tesla K40m, with Compute Capability: 3.5.
```

<a name="2.2"></a>

### 2.2 混合精度O1模式训练

使用如下命令开启单机单卡混合精度O1训练：

```bash
python3 train.py --data-path=./ILSVRC2012 --lr=0.1 --batch-size=256 --amp_level=O1
```

部分训练日志如下：

```
[Epoch 1, iter: 0] top1: 0.06250, top5: 0.12500, lr: 0.10000, loss: 6.64041, avg_reader_cost: 0.73829 sec, avg_batch_cost: 0.86344 sec, avg_samples: 16.0, avg_ips: 18.53046 images/sec.
[Epoch 2, iter: 0] top1: 0.12500, top5: 0.12500, lr: 0.10000, loss: 6.11681, avg_reader_cost: 0.73225 sec, avg_batch_cost: 0.90348 sec, avg_samples: 16.0, avg_ips: 17.70925 images/sec.
```

<a name="2.3"></a>

### 2.3 混合精度O2模式训练

使用如下命令开启单机单卡混合精度O2训练：

```bash
python3 train.py --data-path=./ILSVRC2012 --lr=0.1 --batch-size=256 --amp_level=O2
```

部分训练日志如下：

```
[Epoch 1, iter: 0] top1: 0.06250, top5: 0.18750, lr: 0.10000, loss: 6.73047, avg_reader_cost: 0.81649 sec, avg_batch_cost: 0.92645 sec, avg_samples: 16.0, avg_ips: 17.27027 images/sec.
[Epoch 2, iter: 0] top1: 0.37500, top5: 0.50000, lr: 0.10000, loss: 6.39062, avg_reader_cost: 0.71931 sec, avg_batch_cost: 0.83710 sec, avg_samples: 16.0, avg_ips: 19.11364 images/sec.
```


<a name="3"></a>

## 3. FAQ
