## 1. 推理 Benchmark

### 1.1 软硬件环境

* PP-OCRv3模型推理速度测试采用6148CPU，开启MKLDNN，线程数为10进行测试。

### 1.2 数据集
使用内部数据集进行测试

### 1.3 指标


| Model | Hmean |  Model Size (M) | Time Cost (CPU, ms) |
|-----|-----|--------|----|
| PP-OCR mobile | 50.30% | 8.1 | 356.00  |
| PP-OCR server | 57.00% | 155.1 | 1056.00 |
| PP-OCRv2 | 57.60% | 11.6 | 330.00 |
| PP-OCRv3 | 62.90% | 15.6 | 331.00 |


## 2. 相关使用说明
1. https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/PP-OCRv3_introduction.md
