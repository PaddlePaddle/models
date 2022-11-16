## 1. Train Benchmark

PP-LCNetV2 series model training Benchmark evaluation process can be referred to [PaddleClas-TIPC](https://github.com/paddlepaddle/paddleclas/blob/release%2F2.5/test_tipc/docs/benchmark_train.md)。

## 1. Inference Benchmark

### 1.1 Environment

* The following tests are based on the Intel Xeon Gold 6271C and the OpenVINO 2021.4.2;
* The precision type is FP32;

### 2.2 Dataset

The PP-LCNetv2 series models were evaluated using the ImageNet1k validation dataset.

### 2.3 Metrics

| Model | Latency(ms) |
|:--:|:--:|
| <b>PPLCNetv2_base<b>  | <b>4.32<b> |
| <b>PPLCNetv2_base_ssld<b>  | <b>4.32<b> |

## 3. Instructions

For more information please refer to [PaddleClas-PPLCNetv2](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/models/PP-LCNetV2_en.md).
