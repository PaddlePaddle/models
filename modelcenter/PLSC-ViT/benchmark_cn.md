# 1. 推理Benchmark

## 1.1 软硬件环境

- PLSC-ViT模型推理采用GPU的型号为A100,不同的尺度的模型采用了单机8卡或是4机32卡。

## 1.2 数据集
- 测试使用的数据集为ImageNet.

## 1.3 指标


| Model | Phase | Dataset | gpu | img/sec | Top1 Acc | Official |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B_16_224 |pretrain  |ImageNet2012  |A100*N1C8  |  3583| 0.75196 | 0.7479 |
| ViT-B_16_384 |finetune  | ImageNet2012 | A100*N1C8 | 719 | 0.77972 | 0.7791 |
| ViT-L_16_224 | pretrain | ImageNet21K | A100*N4C32 | 5256 | - | - |  |
|ViT-L_16_384  |finetune  | ImageNet2012 | A100*N4C32 | 934 | 0.85030 | 0.8505 |

# 2. 相关使用说明

https://github.com/PaddlePaddle/PLSC/blob/master/task/classification/vit/README.md
