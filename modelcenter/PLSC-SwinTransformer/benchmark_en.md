# 1. Benchmark

## 1.1 Environment

* 8 A100(40G) on single Node
* CUDA 11.2
* CUDNN 8.1

## 1.2 DataSet
- We train the Swin Transformer on ImageNet.

## 1.3 Benchmark


| Model |DType | Phase | Dataset | gpu | img/sec | Top1 Acc | Official |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Swin-B |FP16 O1|pretrain  |ImageNet2012  |A100*N1C8  |  2155| 0.83362 | 0.835 |
| Swin-B |FP16 O2|pretrain  | ImageNet2012 | A100*N1C8 | 3006 | 0.83223     | 0.835 |

# 2. Reference

https://github.com/PaddlePaddle/PLSC/blob/master/task/classification/swin/README.md
