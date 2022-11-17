## 1. Train Benchmark

PP-LCNet series model training Benchmark evaluation process can be referred to [PaddleClas-TIPC](https://github.com/paddlepaddle/paddleclas/blob/release%2F2.5/test_tipc/docs/benchmark_train.md).

## 2. Inference Benchmark

### 2.1 Environment

* The inference speed of PP-LCNet series model is evaluated based on various hardware platforms including Intel CPU, V100 GPU and SD855.
* The precision type is FP32;
* For Intel CPU,the MKLDNN is turned on;
* For V100 GPU,the TensorRT is turned on;

### 2.2 Dataset

The PP-LCNet series models were evaluated using the ImageNet1k validation dataset.

### 2.3 Metrics

#### 2.3.1 Inference speed based on Intel Xeon Gold 6148  

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

#### 2.3.2 Inference speed based on V100 GPU

| Models        | Latency(ms)<br>bs=1 | Latency(ms)<br/>bs=4 | Latency(ms)<br/>bs=8 |
| :--: | :--:| :--: | :--: |
| PPLCNet_x0_25 | 0.72                         | 1.17                             | 1.71                           |
| PPLCNet_x0_35 | 0.69                         | 1.21                             | 1.82                           |
| PPLCNet_x0_5  | 0.70                         | 1.32                             | 1.94                           |
| PPLCNet_x0_75 | 0.71                         | 1.49                             | 2.19                           |
| PPLCNet_x1_0  | 0.73                         | 1.64                             | 2.53                           |
| PPLCNet_x1_5  | 0.82                         | 2.06                             | 3.12                           |
| PPLCNet_x2_0  | 0.94                         | 2.58                             | 4.08                           |

#### 2.3.3 Inference speed based on SD855

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

## 3. Instructions

For more information please refer to [PaddleClas-PPLCNet](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/models/PP-LCNet_en.md).
