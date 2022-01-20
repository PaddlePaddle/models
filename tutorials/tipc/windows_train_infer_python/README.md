# Windows GPU/CPU 基础训练推理开发文档

# 目录

- [1. 简介](#1)
- [2. Windows GPU/CPU 基础训练推理功能开发与规范](#2)
- [3. Windows GPU/CPU 基础训练推理测试开发与规范](#3)

<a name="1"></a>

## 1. 简介

Windows GPU/CPU 基础训练推理开发过程主要步骤与[《Linux GPU/CPU 基础训练推理开发》](../train_infer_python/README.md)一致。启动命令与Linux GPU相同，由于Windows只支持单卡训练与预测，需要设置环境变量 `set CUDA_VISIBLE_DEVICES=0`。

<a name="2"></a>

## 2. Windows GPU/CPU 基础训练推理功能开发与规范

参考[《Linux GPU/CPU 基础训练推理开发文档》](../train_infer_python/README.md)。

<a name="3"></a>

## 3. Windows GPU/CPU 基础训练推理测试开发与规范

参考[《Linux GPU/CPU 基础训练推理测试开发规范》](../train_infer_python/test_train_infer_python.md)。