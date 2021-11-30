# 飞桨训推一体全流程（TIPC）开发文档

## 1. TIPC简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。飞桨训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）旨在为用户提供所有飞桨模型的训练推理部署打通情况，同时提供一套自动化测试工具，方便用户进行一键测试。本文主要介绍了TIPC测试的接入规范。

<div align="center">
    <img src="tipc_guide.png" width="1000">
</div>

## 2. 接入TIPC

对于已完成训练推理功能开发的模型，可以按照本教程接入TIPC测试。模型训练推理功能开发，请参考[飞桨官方文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)。TIPC接入流程一般分为`开发测试工具`和`撰写测试文档`两步。

### 2.1 开发测试工具

参考样板间和开发文档完成测试工具开发：

- 测试工具样板间：https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph/test_tipc
- 测试工具开发文档：
    - 基础测试开发规范
    
      (1) [Linux GPU/CPU 基础训练推理测试开发规范](./development_specification_docs/train_infer_python.md)
      
      (2) Linux XPU2 基础训练推理测试开发规范 (coming soon)
      
      (3) Linux DCU 基础训练推理测试开发规范 (coming soon)
      
      (4) Linux NPU 基础训练推理测试开发规范 (coming soon)
      
      (5) [Windows GPU 基础训练推理测试开发规范](./development_specification_docs/Windows_train_infer_python.md)
      
      (6) [macOS CPU 基础训练推理测试开发规范](./development_specification_docs/Mac_train_infer_python.md)  
          
    - 更多训练方式测试开发规范

      (1) [多机多卡训练推理测试开发规范](./development_specification_docs/fleet_train_infer_python.md)
      
      (2) [混合精度训练推理测试开发规范](./development_specification_docs/amp_train_infer_python.md)
    
    - 更多部署方式测试开发规范 

      (1) [C++ 推理测试开发规范](./development_specification_docs/inference_cpp.md)
      
      (2) [服务化部署测试开发规范](./development_specification_docs/serving.md)
      
      (3) Paddle.js 部署测试开发规范 (coming soon)
      
      (4) [Paddle2ONNX 测试开发规范](./development_specification_docs/paddle2onnx.md)
      
      (5) [ARM CPU 部署测试开发规范](./development_specification_docs/Lite_arm_cpu_cpp_infer.md)
      
      (6) [OpenCL ARM GPU 部署测试开发规范](./development_specification_docs/Lite_arm_gpu_opencl_cpp_infer.md)

      (7) Metal ARM GPU 部署测试开发规范 (coming soon)
      
      (8) [Jetson 部署测试开发规范](./development_specification_docs/Jeston_infer_python.md)
      
      (9) XPU 部署测试开发规范 (coming soon)
      

### 2.2 撰写测试文档

参考测试文档样板间撰写测试文档：

- 测试文档样板间：https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/test_tipc/readme.md
