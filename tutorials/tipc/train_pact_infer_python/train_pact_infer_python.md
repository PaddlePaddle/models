# Linux GPU/CPU PACT量化训练功能开发文档

# 目录

- [1. 简介](#1)
- [2. 量化训练功能开发](#2)
    - [2.1 准备数据和环境](#2.1)
    - [2.2 准备待量化模型](#2.2)
    - [2.3 准备量化训练代码](#2.3)
    - [2.4 开始量化训练及保存模型](#2.4)
    - [2.5 验证推理结果正确性](#2.5)
- [3. FAQ](#3)
    - [3.1 通用问题](#3.1)


<a name="1"></a>

## 1. 简介

Paddle 量化训练（Quant-aware Training, QAT）是指在训练过程中对模型的权重及激活做模拟量化，并且产出量化训练校准后的量化模型，使用该量化模型进行预测，可以减少计算量、降低计算内存、减小模型大小。

更多关于PaddleSlim 量化的介绍，可以参考[PaddleSlim 量化训练官网教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/dygraph/quanter/qat.rst#%E9%87%8F%E5%8C%96%E8%AE%AD%E7%BB%83)。

<a name="2"></a>

## 2. 量化训练功能开发

Linux GPU/CPU PACT量化训练功能开发可以分为5个步骤，如下图所示。

<div align="center">
    <img src="../images/quant_aware_training_guide.png" width="600">
</div>

其中设置了2个核验点，分别为：

* 准备待量化模型
* 验证量化模型推理结果正确性

<a name="2.1"></a>

### 2.1 准备数据和环境

**【准备校准数据】**

将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

选择适量训练集或验证集

**【准备开发环境】**

- 确定已安装PaddlePaddle最新版本，通过pip安装linux版本paddle命令如下，更多的版本安装方法可查看飞桨[官网](https://www.paddlepaddle.org.cn/)
- 确定已安装paddleslim最新版本，通过pip安装linux版本paddle命令如下，更多的版本安装方法可查看[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)

```
pip install paddlepaddle-gpu
pip install paddleslim
```

<a name="2.2"></a>

### 2.2 准备待量化模型

**【基本流程】**

- Step1：定义继承自`paddle.nn.Layer`的网络模型

**【实战】**

模型组网可以参考[mobilenet_v3](https://github.com/PaddlePaddle/models/blob/release/2.2/tutorials/mobilenetv3_prod/Step6/paddlevision/models/mobilenet_v3.py)

```python
fp32_model = mobilenet_v3_small()
```

<a name="2.3"></a>

### 2.3 准备量化训练代码

**【基本流程】**

PACT在线量化训练开发之前，要求首先有Linux GPU/CPU基础训练的代码并可以正常训练与收敛。

**【实战】**

参考MobileNetV3_small的训练过程说明文档：[链接](https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/mobilenetv3_prod/Step6/README.md#41-%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)。

<a name="2.4"></a>

### 2.4 开始量化训练及保存模型

**【基本流程】**

使用飞桨PaddleSlim中的`QAT`接口开始进行量化训练：

- Step1：配置量化训练参数。

```python
quant_config = {
    'weight_preprocess_type': None,
    'activation_preprocess_type': PACT, #None,
    'weight_quantize_type': 'channel_wise_abs_max',
    'activation_quantize_type': 'moving_average_abs_max',
    'weight_bits': 8,
    'activation_bits': 8,
    'dtype': 'int8',
    'window_size': 10000,
    'moving_rate': 0.9,
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}
```

**注意**：保持以上量化配置，无需改动


- Step2：插入量化算子，得到量化训练模型

```python
from paddleslim.dygraph.quant import QAT
quanter = QAT(config=quant_config)
quanter.quantize(net)
```

- Step3：开始训练。

- Step4：量化训练结束，保存量化模型

```python
quanter.save_quantized_model(net, 'save_dir', input_spec=[paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32')])
```


**【实战】**

量化训练配置、训练及保存量化模型请参考[MobileNetv3量化训练文档](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6/docs/train_pact_infer_python.md)

<a name="2.5"></a>

### 2.5 通过Paddle Inference验证量化前模型和量化后模型的精度差异

**【基本流程】**

可参考[开发推理程序流程](https://github.com/PaddlePaddle/models/blob/release/2.3/tutorials/tipc/train_infer_python/infer_python.md#26-%E5%BC%80%E5%8F%91%E6%8E%A8%E7%90%86%E7%A8%8B%E5%BA%8F)

**【实战】**


1）初始化`paddle.inference`库并配置相应参数：

具体可以参考MobileNetv3 [Inference模型测试代码](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6/deploy/ptq_python/eval.py)

2）配置预测库输入输出：

具体可以参考MobileNetv3 [Inference模型测试代码](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6/deploy/ptq_python/eval.py)

3）开始预测：

具体可以参考MobileNetv3 [Inference模型测试代码](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6/deploy/ptq_python/eval.py)

4）测试单张图像预测结果是否正确，可参考[Inference预测文档](https://github.com/PaddlePaddle/models/blob/release/2.2/docs/tipc/train_infer_python/infer_python.md)

5）同时也可以测试量化模型和FP32模型的精度，确保量化后模型精度损失符合预期。参考[MobileNet量化模型精度验证文档](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6/deploy/ptq_python/README.md)

<a name="3"></a>

## 3. FAQ

### 3.1 通用问题
如果您在使用该文档完成PACT量化训练的过程中遇到问题，可以给在[这里](https://github.com/PaddlePaddle/PaddleSlim/issues)提一个ISSUE，我们会高优跟进。
