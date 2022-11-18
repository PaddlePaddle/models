## 1. Inference Benchmark


### 1.1 Environment

1. 计算卡：T4、CUDA11.2、CuDNN8.2

2. CPU：Intel(R) Xeon(R) Gold 6271C CPU

3. PaddlePaddle Version：2.3

4. PaddleNLP Version：2.3

5. The unit of performance data is QPS. How to calculate QPS: fixed batch size of 32, test running time total_time, calculated QPS = total_samples / total_time.

6. Metrics：Accuracy for sequence classification，F1-Score for token classification, EM (Exact Match) for question answering.

### 1.2 数据集

Dataset：CLUE TNEWS(sequence classofication)、MSRA_NER(token classification)、CLUE CMRC2018(question answering)

### 1.3 Benchmark

##### CPU Performance

The test environment and instructions are as above. When testing the CPU performance, the number of threads is set to 12.

|                            | TNEWS Performance  | TNEWS Accuracy   | MSRA_NER Performance | MSRA_NER F1 Score | CMRC2018 Performance | CMRC2018 EM |
| -------------------------- | ------------ | ------------ | ------------- | ------------- | ------------- | ------------- |
| ERNIE 3.0-Medium+FP32      | 311.95(1.0X) | 57.45        | 90.91(1.0x)   | 93.04         | 33.74(1.0x)   | 66.95         |
| ERNIE 3.0-Medium+INT8      | 600.35(1.9x) | 56.57(-0.88) | 141.00(1.6x)  | 92.64(-0.40)  | 56.51(1.7x)   | 66.23(-0.72)  |
| ERNIE 3.0-Medium+prune+FP32 | 408.65(1.3x) | 57.31(-0.14) | 122.13(1.3x)  | 93.27(+0.23)  | 48.47(1.4x)   | 65.55(-1.40)  |
| ERNIE 3.0-Medium+prune+INT8 | 704.42(2.3x) | 56.69(-0.76) | 215.58(2.4x)  | 92.39(-0.65)  | 75.23(2.2x)   | 63.47(-3.48)  |

After same compression, the speedup ratio of three models reaches about 2.3.

##### GPU Performance

|                            | TNEWS Performance  | TNEWS Accuracy   | MSRA_NER Performance | MSRA_NER F1 Score | CMRC2018 Performance | CMRC2018 EM |
| -------------------------- | ------------- | ------------ | ------------- | ------------- | ------------- | ------------- |
| ERNIE 3.0-Medium+FP32      | 1123.85(1.0x) | 57.45        | 366.75(1.0x)  | 93.04         | 146.84(1.0x)  | 66.95         |
| ERNIE 3.0-Medium+FP16      | 2672.41(2.4x) | 57.45(0.00)  | 840.11(2.3x)  | 93.05(0.01)   | 303.43(2.1x)  | 66.95(0.00)   |
| ERNIE 3.0-Medium+INT8      | 3226.26(2.9x) | 56.99(-0.46) | 889.33(2.4x)  | 92.70(-0.34)  | 348.84(2.4x)  | 66.32(-0.63   |
| ERNIE 3.0-Medium+prune+FP32 | 1424.01(1.3x) | 57.31(-0.14) | 454.27(1.2x)  | 93.27(+0.23)  | 183.77(1.3x)  | 65.92(-1.03)  |
| ERNIE 3.0-Medium+prune+FP16 | 3577.62(3.2x) | 57.27(-0.18) | 1138.77(3.1x) | 93.27(+0.23)  | 445.71(3.0x)  | 65.89(-1.06)  |
| ERNIE 3.0-Medium+prune+INT8 | 3635.48(3.2x) | 57.26(-0.19) | 1105.26(3.0x) | 93.20(+0.16)  | 444.27(3.0x)  | 66.17(-0.78)  |

The three tasks have a speedup of about 3 times after pruning and quantization, and the average accuracy loss could be controlled within 0.5 (0.46).

## 2. Reference
1. https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/ernie-3.0/README.md#%E6%80%A7%E8%83%BD%E6%B5%8B%E8%AF%95
