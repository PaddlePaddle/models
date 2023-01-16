# 1. Benchmark

## 1.1 Environment

- We train the ViT on 1 node with 8 A100 gpus or 4 nodes with 32 A100 gpus.

## 1.2 DataSet
- We train the ViT on ImageNet.

## 1.3 Benchmark


| Model | Phase | Dataset | gpu | img/sec | Top1 Acc | Official |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B_16_224 |pretrain  |ImageNet2012  |A100*N1C8  |  3583| 0.75196 | 0.7479 |
| ViT-B_16_384 |finetune  | ImageNet2012 | A100*N1C8 | 719 | 0.77972 | 0.7791 |
| ViT-L_16_224 | pretrain | ImageNet21K | A100*N4C32 | 5256 | - | - |  |
|ViT-L_16_384  |finetune  | ImageNet2012 | A100*N4C32 | 934 | 0.85030 | 0.8505 |

# 2. Reference

https://github.com/PaddlePaddle/PLSC/blob/master/task/classification/vit/README.md
