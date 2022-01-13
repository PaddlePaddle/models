# Linux GPU/CPU PACT量化训练开发文档

# 目录

- [1. 简介](#1)
- [2. Linux GPU/CPU PACT量化训练功能开发规范](#2)
    - [2.1 开发流程](#2.1)
    - [2.2 核验点](#2.2)
- [3. Linux GPU/CPU PACT量化训练测试开发与规范](#3)
    - [3.1 开发流程](#3.1)
    - [3.2 核验点](#3.2)


<a name="1"></a>

## 1. 简介

该系列文档主要介绍 Linux GPU/CPU PACT量化训练开发过程，主要包含2个步骤。


- 步骤一：参考[《Linux GPU/CPU PACT量化训练功能开发文档》](./train_pact_infer_python.md)，完成Linux GPU/CPU PACT量化训练功能开发。

- 步骤二：参考[《Linux GPU/CPU PACT量化训练测试开发文档》](./test_train_pact_infer_python.md)，完成Linux GPU/CPU PACT量化训练测试开发。


<a name="2"></a>

# 2. Linux GPU/CPU PACT量化训练功能开发规范

<a name="2.1"></a>

### 2.1 开发流程

Linux GPU/CPU PACT量化训练功能开发过程可以分为下面5个步骤。

<div align="center">
    <img src="../images/quant_aware_training_guide.png" width="800">
</div>


更多的介绍可以参考：[Linux GPU/CPU PACT量化训练功能开发文档](././train_pact_infer_python.md)。

<a name="2.2"></a>

### 2.2 核验点

#### 2.2.1 准备待量化模型

* 需要定义继承自`paddle.nn.Layer`的网络模型，该模型与Linux GPU/CPU基础训练过程一致。定义完成之后，建议加载预训练模型权重，加速量化收敛。

#### 2.2.2 验证推理结果正确性

* 使用Paddle Inference库测试离线量化模型，确保模型精度符合预期。对于CV任务来说，PACT在线量化之后的精度基本无损。

<a name="3"></a>

# 3. Linux GPU/CPU PACT量化训练测试开发与规范

coming soon!
