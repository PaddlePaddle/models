# 飞桨训推一体全流程（TIPC）测试开发文档

## 1. TIPC简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。飞桨训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）旨在建立模型从学术研究到产业落地的桥梁，方便模型更广泛的使用。

<div align="center">
    <img src="tipc_guide.png" width="1000">
</div>

## 2. 不同环境不同训练推理方式的测试开发文档 

- [Linux GPU/CPU 基础训练推理测试开发文档](./development_specification_docs/train_infer_python.md)
     
- 更多训练方式开发文档
    - [Linux GPU 多机多卡训练推理测试开发文档](./development_specification_docs/train_fleet_infer_python.md)
    - [Linux GPU 混合精度训练推理测试开发文档](./development_specification_docs/train_amp_infer_python.md)
    
- 更多部署方式开发文档
    - [Linux GPU/CPU C++ 推理测试开发文档](./development_specification_docs/infer_cpp.md)
    - [Linux GPU/CPU 服务化部署测试开发文档](./development_specification_docs/serving.md)
    - Paddle.js 部署测试开发文档 (coming soon)
    - [Paddle2ONNX 测试开发文档](./development_specification_docs/paddle2onnx.md)
    - [ARM CPU 部署测试开发文档](./development_specification_docs/lite_infer_cpp_arm_cpu.md)
    - [OpenCL ARM GPU 部署测试开发文档](./development_specification_docs/lite_infer_cpp_arm_gpu_opencl.md)
    - Metal ARM GPU 部署测试开发文档 (coming soon)
    - [Jetson 部署测试开发文档](./development_specification_docs/infer_python_jeston.md)
    - XPU 部署测试开发文档 (coming soon)

- 更多训练环境开发文档
    - Linux XPU2 基础训练推理测试开发文档 (coming soon)
    - Linux DCU 基础训练推理测试开发文档 (coming soon)
    - Linux NPU 基础训练推理测试开发文档 (coming soon)
    - [Windows GPU 基础训练推理测试开发文档](./development_specification_docs/windows_train_infer_python.md)
    - [macOS CPU 基础训练推理测试开发文档](./development_specification_docs/macos_train_infer_python.md)
