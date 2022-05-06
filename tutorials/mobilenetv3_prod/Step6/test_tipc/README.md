
# 飞桨训推一体全流程（TIPC）

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了飞桨训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

## 2. 汇总信息

打通情况汇总如下，已填写的部分表示可以使用本工具进行一键测试，未填写的表示正在支持中。

**字段说明：**
- 基础训练预测：指Linux GPU/CPU环境下的模型训练、Paddle Inference Python预测。
- 更多训练方式：包括多机多卡、混合精度训练。
- 更多部署方式：包括C++预测、Serving服务化部署、ARM端侧部署等多种部署方式，具体列表见[3.3节](#3.3)
- Slim训练部署：包括PACT在线量化、离线量化。
- 更多训练环境：包括Windows GPU/CPU、Linux NPU、Linux DCU等多种环境。

| 算法论文 | 模型名称 | 模型类型 | 基础<br>训练预测 | 更多<br>训练方式 | 更多<br>部署方式 | Slim<br>训练部署 |  更多<br>训练环境  |
| :--- | :--- |  :----:  | :--------: |  :----:  |   :----:  |   :----:  |   :----:  |
| MobileNetV3     | mobilenet_v3_small |  分类  | 支持 | 混合精度 | PYTHON 服务化部署<br>Paddle2ONNX 部署| PACT量化<br>离线量化 | Windows GPU/CPU |


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

<a name="3.3"></a>
### 3.3 开始测试

请参考相应文档，完成指定功能的测试。

- 基础训练预测测试：
    - [Linux GPU/CPU 基础训练推理测试](docs/test_train_inference_python.md)

- 更多训练方式测试：
    - [Linux GPU/CPU 多机多卡训练推理测试](docs/test_train_fleet_inference_python.md)
    - [Linux GPU/CPU 混合精度训练推理测试](docs/test_train_amp_inference_python.md)

- 更多部署方式测试（coming soon）：
    - [Linux GPU/CPU PYTHON 服务化部署测试](docs/test_serving_infer_python.md)
    - [Linux GPU/CPU C++ 服务化部署测试]
    - [Linux GPU/CPU C++ 推理测试](docs/test_inference_cpp.md)
    - [Paddle.js 部署测试]
    - [Paddle2ONNX 测试](docs/test_paddle2onnx.md)
    - [Lite ARM CPU 部署测试](docs/test_lite_infer_cpp_arm_cpu.md)
    - [OpenCL ARM GPU 部署测试]
    - [Metal ARM GPU 部署测试]
    - [Jetson 部署测试]
    - [XPU 部署测试]
    - [OpenCL ARM GPU 部署测试]

- Slim训练部署测试：
    - [Linux GPU/CPU PACT量化训练测试](./docs/test_train_pact_inference_python.md)
    - [Linux GPU/CPU 离线量化测试](./docs/test_train_ptq_inference_python.md)

- 更多训练环境测试（coming soon）：
    - [Linux XPU2 基础训练推理测试]
    - [Linux DCU 基础训练推理测试]
    - [Linux NPU 基础训练推理测试]
    - [Windows GPU/CPU 基础训练推理测试](docs/test_windows_train_inference_python.md)
