## 1. Inference Benchmark

### 1.1 Environment

The PP-OCRv2 model inference speed test uses 6148CPU, MKLDNN is turned on, and the number of threads is 10 for testing.

### 1.2 Benchmark


| model                | hardware           | det preprocess time | det inference time | det post process time | rec preprocess time | rec inference time | rec post process time | total time (s)    |
|---|---|---|---|---|---|---|---|---|
| ppocr_mobile | 6148CPU      | 10.6291             | 103.8162           | 16.0715               | 0.246               | 62.8177            | 4.6695                | 40.4602 + 69.9684 |
| ppocr_server | 6148CPU      | 10.6834             | 178.5214           | 16.2959               | 0.2741              | 237.5255           | 4.8711                | 63.7052 + 263.783 |
| ppocr_mobile_v2 | 6148CPU      | 10.58               | 102.9626           | 16.5514               | 0.2418              | 53.395             | 4.4622                | 40.3293 + 62.2241 |


## 2. Reference
1. https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.4#pp-ocrv2-pipeline
