# 飞桨训推一体全流程（TIPC）开发文档

## 1. TIPC简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。飞桨训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）旨在建立模型从学术研究到产业落地的桥梁，方便模型更广泛的使用。

<div align="center">
    <img src="images/tipc_guide.png" width="800">
</div>

## 2. 不同环境不同训练推理方式的开发文档

- [Linux GPU/CPU 基础训练推理开发文档](./train_infer_python/README.md)

- 更多训练方式开发文档
    - Linux GPU 多机多卡训练推理开发文档(coming soon)
    - [Linux GPU 混合精度训练推理开发文档](./train_amp_infer_python/README.md)

- 更多部署方式开发文档
    - [Linux GPU/CPU PYTHON 服务化部署开发文档](./serving_python/README.md)
    - Linux GPU/CPU C++ 服务化部署开发文档 (coming soon)
    - Linux GPU/CPU C++ 推理开发文档 (coming soon)
    - Paddle.js 部署开发文档 (coming soon)
    - [Paddle2ONNX 开发文档](./paddle2onnx/README.md)
    - [Lite ARM CPU 部署开发文档](./lite_infer_cpp_arm_cpu/README.md)
    - OpenCL ARM GPU 部署开发文档 (coming soon)
    - Metal ARM GPU 部署开发文档 (coming soon)
    - Jetson 部署开发文档 (coming soon)
    - XPU 部署开发文档 (coming soon)

- Slim训练部署开发文档
    - [Linux GPU/CPU PACT量化训练开发文档](./train_pact_infer_python/README.md)
    - [Linux GPU/CPU 离线量化开发文档](./ptq_infer_python/README.md)

- 更多训练环境开发文档
    - Linux XPU2 基础训练推理开发文档 (coming soon)
    - [Linux DCU 基础训练推理开发文档](./linux_dcu_train_infer_python/README.md)
    - Linux NPU 基础训练推理开发文档 (coming soon)
    - [Windows GPU/CPU 基础训练推理开发文档](./windows_train_infer_python/README.md)
