
# 飞桨训推一体全流程（TIPC）

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了mobilenet_v3中所有模型的飞桨训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

<div align="center">
    <img src="docs/tipc_guide.png" width="1000">
</div>

## 2. 汇总信息

打通情况汇总如下，已填写的部分表示可以使用本工具进行一键测试，未填写的表示正在支持中。

**字段说明：**
- 基础训练预测：包括模型训练、Paddle Inference Python预测。
- 更多训练方式：包括多机多卡、混合精度。
- 模型压缩：包括裁剪、离线/在线量化、蒸馏。
- 其他预测部署：包括Paddle Inference C++预测、Paddle Serving部署、Paddle-Lite部署等。

更详细的MKLDNN、Tensorrt等预测加速相关功能的支持情况可以查看各测试工具的[更多教程](#more)。

| 算法论文 | 模型名称 | 模型类型 | 基础<br>训练预测 | 更多<br>训练方式 | 模型压缩 |  其他预测部署  |
| :--- | :--- |  :----:  | :--------: |  :----  |   :----  |   :----  |
| MobileNetV3     | mobilenet_v3_small | 分类  | 支持 | - | - | Paddle Serving: Python |

## 3. 测试工具简介

### 3.1 目录介绍

```shell
test_tipc/
├── configs/  # 配置文件目录
    ├── mobilenet_v3_small            # 模型的测试配置文件目录
        ├── train_infer_python.txt    # 测试Linux GPU/CPU 基础训练推理测试的配置文件
├── test_train_inference_python.sh    # 测试python训练预测的主程序 (无需修改)
├── test_serving.sh                   # 测试serving部署预测的主程序 (无需修改)
├── common_func.sh                    # 测试过程中使用的通用函数（无需修改）
└── README.md                         # 使用文档
```

### 3.2 测试流程概述

使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐。测试过程包含

1. 准备数据与环境
2. 运行测试脚本，观察不同配置是否运行成功。

具体内容可以参考第4章中的测试文档链接。

<a name="more"></a>

## 4. 更多测试功能

更多测试功能可以参考

* [Linux GPU/CPU 基础训练推理测试文档](docs/test_train_inference_python.md)
* [PaddleServing 测试文档](docs/test_serving.md)
