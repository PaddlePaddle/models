## 1. 训练Benchmark

### 1.1 软硬件环境

* PP-TSM模型训练过程中使用8 GPUs，每GPU batch size为16进行训练，如训练GPU数和batch size不使用上述配置，需要线性调整学习率和迭代次数。

### 1.2 数据集

PP-TSM模型使用Kinetics-400数据集进行训练和测试。

### 1.3 指标

|模型名称 | 模型简介 | 输入尺寸 | 输入帧数 | ips |
|---|---|---|---|---|
|pptsm_k400_frames_uniform | 行为识别 | 224 | 8 | 274.32 |



## 2. 推理 Benchmark

### 2.1 软硬件环境

* PP-TSM模型推理速度测试采用单卡V100，batch size=1进行测试，使用CUDA 10.2, CUDNN 8.1.1，TensorRT推理速度测试使用TensorRT 7.0.0.11。


### 2.2 数据集

PP-TSM模型使用Kinetics-400数据集进行训练和测试。

### 2.3 指标

|模型名称 | 精度% | 预处理时间ms | 模型推理时间ms | 预测总时间ms |
| :---- | :----: | :----: |:----: |:----: |
|pptsm_k400_frames_uniform | 75.11 | 51.84 | 11.26 | 63.1 |

## 3. 参考

参考文档: https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/benchmark.md