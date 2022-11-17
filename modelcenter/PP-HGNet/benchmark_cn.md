## 1. 训练 Benchmark

PP-HGNet 系列模型的训练 Benchmark 评测流程可以参考 [PaddleClas-TIPC](https://github.com/paddlepaddle/paddleclas/blob/release%2F2.5/test_tipc/docs/benchmark_train.md)。

## 2. 推理 Benchmark

### 2.1 软硬件环境

* 下列测试基于 NVIDIA® Tesla® V100 硬件平台，开启 TensorRT 加速；
* 下列测试均基于 FP32 精度完成；

### 2.2 数据集

PP-HGNet 系列模型使用 ImageNet1k validation 数据集进行评测。

### 2.3 指标

| Model | Latency(ms) |
|:--: |:--: |
| PPHGNet_tiny      | 1.77 |
| PPHGNet_tiny_ssld  | 1.77 |
| PPHGNet_small     | 2.52  |
| PPHGNet_small_ssld | 2.52  |
| PPHGNet_base_ssld | 5.97   |


## 3. 相关使用说明

更多信息请参考[PaddleClas-PPHGNet](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/PP-HGNet.md)。
