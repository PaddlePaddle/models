## 1. 推理 Benchmark

### 1.1 软硬件环境

* 下列测试基于 Intel Xeon Gold 6271C 硬件平台与 OpenVINO 2021.4.2 推理平台完成；
* 下列测试均基于 FP32 精度完成；

### 2.2 数据集

PP-LCNetv2 系列模型使用 ImageNet1k validation 数据集进行评测。

### 2.3 指标

| Model | Latency(ms) |
|:--:|:--:|
| <b>PPLCNetv2_base<b>  | <b>4.32<b> | 
| <b>PPLCNetv2_base_ssld<b>  | <b>4.32<b> |

## 3. 相关使用说明

更多信息请参考[PaddleClas-PPLCNetv2](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/PP-LCNetV2.md)。
