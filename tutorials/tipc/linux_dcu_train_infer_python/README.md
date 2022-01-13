# Linux DCU 基础训练推理开发文档

# 目录

- [1. 简介](#1)
- [2. Linux DCU 基础训练推理功能开发与规范](#2)
- [3. Linux DCU 基础训练推理测试开发与规范](#3)

<a name="1"></a>

## 1. 简介

Linux DCU 基础训练推理开发过程主要步骤与[《Linux GPU/CPU 基础训练推理开发》](../train_infer_python/README.md)一致。启动命令与Linux GPU相同，只需修改环境变量 `CUDA_VISIBLE_DEVICES` 为 `HIP_VISIBLE_DEVICES`。

<a name="2"></a>

## 2. Linux DCU 基础训练推理功能开发与规范

参考[《Linux GPU/CPU 基础训练推理开发文档》](../train_infer_python/README.md)。

<a name="3"></a>

## 3. Linux DCU 基础训练推理测试开发与规范

参考[《Linux GPU/CPU 基础训练推理测试开发规范》](../train_infer_python/test_train_inference_python.md)。
