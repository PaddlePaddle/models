## 1. Train Benchmark

PP-HGNet series model training Benchmark evaluation process can be referred to [PaddleClas-TIPC](https://github.com/paddlepaddle/paddleclas/blob/release%2F2.5/test_tipc/docs/benchmark_train.md)。

## 1. Inference Benchmark

### 1.1 Environment

* The following tests are based on the NVIDIA® Tesla® V100,and the TensorRT engine is turned on;
* The precision type is FP32;

### 2.2 Dataset

The PP-HGNet series models were evaluated using the ImageNet1k validation dataset.

### 2.3 Metrics

| Model | Latency(ms) |
|:--: |:--: |
| PPHGNet_tiny      | 1.77 |
| PPHGNet_tiny_ssld  | 1.77 |
| PPHGNet_small     | 2.52  |
| PPHGNet_small_ssld | 2.52  |
| PPHGNet_base_ssld | 5.97   |


## 3. Instructions

For more information please refer to[PaddleClas-PPHGNet](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/PP-HGNet.md)。
