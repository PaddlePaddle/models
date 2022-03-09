# Windows GPU/CPU 基础训练推理开发文档

# 目录

- [1. 简介](#1)
- [2. Windows GPU/CPU 基础训练推理功能开发与规范](#2)
- [3. Windows GPU/CPU 基础训练推理测试开发与规范](#3)

<a name="1"></a>

## 1. 简介

Windows GPU/CPU 基础训练推理开发过程主要步骤与[《Linux GPU/CPU 基础训练推理开发》](../train_infer_python/README.md)一致，对应的mobilenet_v3_small模型示例参考[MobileNetV3](https://github.com/PaddlePaddle/models/blob/release/2.2/tutorials/mobilenetv3_prod/Step6/docs/windows_train_infer_python.md)。

**注意事项：**
* 由于Windows只支持单卡训练与预测，需要设置环境变量 `set CUDA_VISIBLE_DEVICES=0`。
* Python在Windows上int数据默认为int32类型，在调用某些函数时会报错："it holds int, but desires to be int64_t。此时需要显式调用astype("int64")，将输入转换为int64类型。
* 在Windows平台，DataLoader只支持单进程模式，因此需要设置 workers 为0。

<a name="2"></a>

## 2. Windows GPU/CPU 基础训练推理功能开发与规范

参考[《Linux GPU/CPU 基础训练推理开发文档》](../train_infer_python/README.md)。

<a name="3"></a>

## 3. Windows GPU/CPU 基础训练推理测试开发与规范

[《Windows GPU/CPU 基础训练推理测试开发规范》](./test_windows_train_infer_python.md)。
