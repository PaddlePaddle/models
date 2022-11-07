## 1. Inference Benchmark

### 1.1 Environment

The PP-OCRv3 model inference speed test uses 6148CPU, MKLDNN is turned on, and the number of threads is 10 for testing.

### 1.2 Benchmark


| Model | Hmean |  Model Size (M) | Time Cost (CPU, ms) | Time Cost (T4 GPU, ms) |
|-----|-----|--------|----| --- |
| PP-OCR mobile | 50.30% | 8.1 | 356.00  | 116.00 |
| PP-OCR server | 57.00% | 155.1 | 1056.00 | 200.00 |
| PP-OCRv2 | 57.60% | 11.6 | 330.00 | 111.00 |
| PP-OCRv3 | 62.90% | 15.6 | 331.00 | 86.64 |

## 2. Reference
1. https://github.com/PaddlePaddle/PaddleOCR/edit/dygraph/doc/doc_ch/PP-OCRv3_introduction.md
