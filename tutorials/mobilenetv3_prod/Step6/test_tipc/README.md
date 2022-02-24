
# 飞桨训推一体全流程（TIPC）

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了飞桨训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

## 2. 汇总信息

打通情况汇总如下，已填写的部分表示可以使用本工具进行一键测试，未填写的表示正在支持中。

**字段说明：**
- 基础训练预测：包括模型训练、Paddle Inference Python预测。
- 更多训练方式：包括多机多卡、混合精度。
- 模型压缩：包括裁剪、离线/在线量化、蒸馏。
- 其他预测部署：包括Paddle Inference C++预测、Paddle Serving部署、Paddle-Lite部署等。

| 算法论文 | 模型名称 | 模型类型 | 基础<br>训练预测 | 更多<br>训练方式 | 模型压缩 |  其他预测部署  |
| :--- | :--- |  :----:  | :--------: |  :----  |   :----  |   :----  |
| MobileNetV3     | mobilenet_v3_small |  分类  | 支持 | - | - | - |


## 3. 测试工具简介

### 3.1 目录介绍

```
test_tipc
    |--configs                              # 配置目录
    |    |--model_name                      # 您的模型名称
    |           |--train_infer_python.txt   # 基础训练推理测试配置文件
    |--docs                                 # 文档目录
    |   |--test_train_inference_python.md   # 基础训练推理测试说明文档
    |----README.md                          # TIPC说明文档
    |----prepare.sh                         # TIPC基础训练推理测试数据准备脚本
    |----test_train_inference_python.sh     # TIPC基础训练推理测试解析脚本，无需改动
    |----common_func.sh                     # TIPC基础训练推理测试常用函数，无需改动
```

### 3.2 测试流程概述

使用本工具，可以测试不同功能的支持情况。测试过程包含：

1. 准备数据与环境
2. 运行测试脚本，观察不同配置是否运行成功。

### 3.3 开始测试

请参考相应文档，完成指定功能的测试。

- 基础训练预测测试：
    - [Linux GPU/CPU 基础训练推理测试](docs/test_train_inference_python.md)

- 更多训练方式测试（coming soon）：
    - [Linux GPU/CPU 多机多卡训练推理测试]
    - [Linux GPU/CPU 混合精度训练推理测试]

- 更多部署方式测试（coming soon）：
    - [Linux GPU/CPU PYTHON 服务化部署测试]
    - [Linux GPU/CPU C++ 服务化部署测试]
    - [Linux GPU/CPU C++ 推理测试]
    - [Paddle.js 部署测试]
    - [Paddle2ONNX 测试]
    - [Lite ARM CPU 部署测试]
    - [OpenCL ARM GPU 部署测试]
    - [Metal ARM GPU 部署测试]
    - [Jetson 部署测试]
    - [XPU 部署测试]
    - [OpenCL ARM GPU 部署测试]

- Slim训练部署测试（coming soon）：
    - [Linux GPU/CPU PACT量化训练测试]
    - [Linux GPU/CPU 离线量化测试]

- 更多训练环境测试（coming soon）：
    - [Linux XPU2 基础训练推理测试]
    - [Linux DCU 基础训练推理测试]
    - [Linux NPU 基础训练推理测试]
    - [Windows GPU/CPU 基础训练推理测试]
