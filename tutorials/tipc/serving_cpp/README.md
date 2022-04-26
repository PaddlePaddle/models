# Linux GPU/CPU C++ 服务化部署开发文档

# 目录

- [1. 简介](#1)
- [2. 服务化部署功能开发与规范](#2)
    - [2.1 开发流程](#2.1)
    - [2.2 核验点](#2.2)
- [3. 服务化部署测试开发与规范](#3)
    - [3.1 开发流程](#3.1)
    - [3.2 核验点](#3.2)

<a name="1"></a>

## 1.简介
该系列文档主要介绍 Linux GPU/CPU C++ 服务化部署开发过程，主要包含2个步骤。
- 步骤一：参考[《Linux GPU/CPU C++ 服务化部署功能开发文档》](./serving_cpp.md)，完成Linux GPU/CPU C++ 服务化部署功能开发。
- 步骤二：参考[《Linux GPU/CPU C++ 服务化部署功能测试开发文档》](./test_serving_cpp.md)，完成Linux GPU/CPU C++ 服务化部署功能测试开发。

<a name="2"></a>

## 2.服务化部署流程开发规范

<a name="2.1"></a>

### 2.1 开发流程

Linux GPU/CPU C++ 服务化部署功能开发过程以分为下面8个步骤。

<div align="center">
    <img src="../images/serving_guide.png" width="800">
</div>

更多内容请参考：[Linux GPU/CPU C++ 服务化部署功能开发文档]()。

<a name="2.2"></a>

### 2.2 检验点

在开发过程中，至少需要产出下面的内容。

#### 2.2.1 模型服务部署成功

- 成功启动模型预测服务，并在客户端完成访问，返回结果。

#### 2.2.2 服务化部署结果正确性

- 返回结果与基于Paddle Inference的模型推理结果完全一致。

<a name="3"></a>

## 3.服务化部署测试开发规范

<a name="3.1"></a>

### 3.1 开发规范

基础训练推理测试开发的流程如下所示。

<div align="center">
    <img src="../train_infer_python/images/test_linux_train_infer_python_pipeline.png" width="400">
</div>

更多的介绍可以参考：[Linux GPU/CPU C++ 服务化部署测试开发文档]()。

<a name="3.2"></a>

### 3.2 核验点

#### 3.2.1 目录结构

在 repo 根目录下面新建 test_tipc 文件夹，目录结构如下所示。
```
test_tipc
    |--configs                              # 配置目录
    |    |--model_name                      # 您的模型名称
    |           |--serving_infer_python.txt   # python服务化部署测试配置文件
    |--docs                                 # 文档目录
    |   |--test_serving_infer_python.md   # python服务化部署测试说明文档
    |----README.md                          # TIPC说明文档
    |----test_serving_infer_python.sh     # TIPC python服务化部署解析脚本，无需改动
    |----common_func.sh                     # TIPC基础训练推理测试常用函数，无需改动
```

#### 3.2.2 配置文件和测试文档

- `test_tipc/README.md` 文档中对该模型支持的的功能进行总体介绍。
- `test_tipc/docs/test_serving_infer_python.md` 文档中对PaddleServing的功能支持情况进行介绍。
- 根据测试文档，基于配置文件，跑通训练推理全流程测试。
