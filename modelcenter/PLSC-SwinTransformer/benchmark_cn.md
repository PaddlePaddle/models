# 1. Benchmark

## 1.1 软硬件环境

- PLSC-SwinTransformer模型推理采用GPU的型号为A100(40G),采用了单机8卡。

## 1.2 数据集
- 测试使用的数据集为ImageNet.

## 1.3 指标


| Model |DType | Phase | Dataset | gpu | img/sec | Top1 Acc | Official |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Swin-B |FP16 O1|pretrain  |ImageNet2012  |A100*N1C8  |  2155| 0.83362 | 0.835 |
| Swin-B |FP16 O2|pretrain  | ImageNet2012 | A100*N1C8 | 3006 | 0.83223	 | 0.835 |

# 2. 相关使用说明

https://github.com/PaddlePaddle/PLSC/blob/master/task/classification/swin/README.md
