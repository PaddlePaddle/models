## 1. 训练 Benchmark

PP-LCNet 系列模型的训练 Benchmark 评测流程可以参考 [PaddleClas-TIPC](https://github.com/paddlepaddle/paddleclas/blob/release%2F2.5/test_tipc/docs/benchmark_train.md)。

## 1. 推理 Benchmark

### 1.1 软硬件环境

* PP-LCNet 系列模型基于 Intel CPU、V100 GPU、SD855 在内的多种硬件平台对推理速度进行了评测；
* 下列测试均基于 FP32 精度完成；
* 对于 Intel CPU 硬件平台测试，开启 MKLDNN 加速；
* 对于 V100 GPU 硬件平台测试，开启 TensorRT 加速；

### 2.2 数据集

PP-LCNet 系列模型使用 ImageNet1k validation 数据集进行评测。

### 2.3 指标

#### 2.3.1 基于 Intel Xeon Gold 6148 的预测速度  

| Model | Latency(ms)<br/>bs=1, thread=10 |
|:--:|:--:|
| PPLCNet_x0_25  | 1.74 |
| PPLCNet_x0_35  | 1.92 |
| PPLCNet_x0_5   | 2.05 |
| PPLCNet_x0_75  | 2.29 |
| PPLCNet_x1_0   | 2.46 |
| PPLCNet_x1_5   | 3.19 |
| PPLCNet_x2_0   | 4.27 |
| PPLCNet_x2_5   | 5.39 |

#### 2.3.2 基于 V100 GPU 的预测速度

| Models        | Latency(ms)<br>bs=1 | Latency(ms)<br/>bs=4 | Latency(ms)<br/>bs=8 |
| :--: | :--:| :--: | :--: |
| PPLCNet_x0_25 | 0.72                         | 1.17                             | 1.71                           |
| PPLCNet_x0_35 | 0.69                         | 1.21                             | 1.82                           |
| PPLCNet_x0_5  | 0.70                         | 1.32                             | 1.94                           |
| PPLCNet_x0_75 | 0.71                         | 1.49                             | 2.19                           |
| PPLCNet_x1_0  | 0.73                         | 1.64                             | 2.53                           |
| PPLCNet_x1_5  | 0.82                         | 2.06                             | 3.12                           |
| PPLCNet_x2_0  | 0.94                         | 2.58                             | 4.08                           |

#### 2.3.3 基于 SD855 的预测速度

| Models        | Latency(ms)<br>bs=1, thread=1 | Latency(ms)<br/>bs=1, thread=2 | Latency(ms)<br/>bs=1, thread=4 |
| :--: | :--: | :--: | :--: |
| PPLCNet_x0_25 | 2.30                             | 1.62                              | 1.32                              |
| PPLCNet_x0_35 | 3.15                             | 2.11                              | 1.64                              |
| PPLCNet_x0_5  | 4.27                             | 2.73                              | 1.92                              |
| PPLCNet_x0_75 | 7.38                             | 4.51                              | 2.91                              |
| PPLCNet_x1_0  | 10.78                            | 6.49                              | 3.98                              |
| PPLCNet_x1_5  | 20.55                            | 12.26                             | 7.54                              |
| PPLCNet_x2_0  | 33.79                            | 20.17                             | 12.10                             |
| PPLCNet_x2_5  | 49.89                            | 29.60                             | 17.82                             |

## 3. 相关使用说明

更多信息请参考[PaddleClas-PPLCNet](https://github.com/paddlepaddle/paddleclas/blob/release%2F2.5/docs/zh_CN/models/ImageNet1k/PP-LCNet.md)。