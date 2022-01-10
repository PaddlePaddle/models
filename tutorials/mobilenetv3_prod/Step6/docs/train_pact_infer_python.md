# MobileNetV3

## 目录


- [1. 简介](#1)
- [2. PACT量化训练](#2)
    - [2.1 检查环境](#2.1)
    - [2.2 开始量化训练](#2.2)
    - [2.3 验证量化模型指标](#2.3)
- [3. FAQ](#3)


<a name="1"></a>

## 1. 简介

Paddle 量化训练（Quant-aware Training, QAT）是指在训练过程中对模型的权重及激活做模拟量化，并且产出量化训练校准后的量化模型，使用该量化模型进行预测，可以减少计算量、降低计算内存、减小模型大小。

本文档主要基于Paddle的MobileNetV3模型进量化训练。

更多关于PaddleSlim 量化的介绍，可以参考[PaddleSlim 量化训练官网教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/dygraph/quanter/qat.rst#%E9%87%8F%E5%8C%96%E8%AE%AD%E7%BB%83)。


<a name="2"></a>

## 2. PACT量化训练

<a name="2.1"></a>

### 2.1 准备环境

- PACT量化训练依赖于PaddleSlim，需要事先安装PaddlePaddle和PaddleSlim：

```shell
pip install paddlepaddle-gpu==2.2.0
pip install paddleslim==2.2.1
```

- 数据准备请参考[数据准备文档](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6#32-%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE)。

- 准备训好的FP32模型
可以通过[模型训练文档](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6#41-%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83) 准备好训好的模型权重，也可以直接下载预训练模型：
```shell
# 下载预训练模型
wget https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams
```

<a name="2.2"></a>

### 2.2 开始量化训练

使用如下命令开启单机单卡混合精度O1训练：

```bash
python3 train.py --data-path=./ILSVRC2012 \
                 --lr=0.001 --batch-size=64 \
                 --epochs=10 \
                 --pretrained=mobilenet_v3_small_pretrained.pdparams \
                 --pact_quant
```

部分训练日志如下：

```
[Epoch 0, iter: 0] top1: 0.43750, top5: 0.73438, lr: 0.00100, loss: 2.49211, avg_reader_cost: 2.37156 sec, avg_batch_cost: 3.47348 sec, avg_samples: 64.0, avg_ips: 18.42531 images/sec.
[Epoch 0, iter: 10] top1: 0.48594, top5: 0.70625, lr: 0.00100, loss: 2.24239, avg_reader_cost: 0.00026 sec, avg_batch_cost: 0.41364 sec, avg_samples: 64.0, avg_ips: 154.72471 images/sec.
[Epoch 0, iter: 20] top1: 0.45781, top5: 0.69531, lr: 0.00100, loss: 2.36400, avg_reader_cost: 0.00056 sec, avg_batch_cost: 0.42063 sec, avg_samples: 64.0, avg_ips: 152.15310 images/sec.
```

<a name="2.3"></a>

### 2.3 验证量化模型指标

训练完成量化模型，会在`output_dir`路径下生成`qat_inference.pdmodel` 和 `qat_inference.pdiparams` 的Inference模型，可以直接使用Paddle Inference进行预测部署，或者导出Paddle Lite格式进行部署。

为了验证量化后的模型精度或指标，可以参考[量化模型精度验证文档](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6/deploy/ptq_python#23-%E9%AA%8C%E8%AF%81%E6%8E%A8%E7%90%86%E7%BB%93%E6%9E%9C)进行指标或模型效果的验证。

** 注意：** 需要将`--model_filename`指定为`qat_inference.pdmodel`，将`--params_filename`指定为`qat_inference.pdiparams`。

<a name="3"></a>

## 3. FAQ
