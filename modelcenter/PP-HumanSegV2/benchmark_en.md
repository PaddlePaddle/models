## 1. Inference Benchmark

### 1.1 Environment

* Segmentation accuracy (mIoU): We tested the models on PP-HumanSeg-14K dataset using the best input shape, without tricks like multi-scale and flip.
* Inference latency on Snapdragon 855 ARM CPU: We tested the models on xiaomi9 (Snapdragon 855 CPU) using PaddleLite, with single thread, large kernel and best input shape.
* Inference latency on Intel CPU: We tested the models on Intel(R) Xeon(R) Gold 6271C CPU using Paddle Inference.

### 1.2 Datasets

* PP-HumanSeg14k[(paper)](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/paper.md), proposed by PaddleSeg team, is used for testing.

### 1.3 Benchmark

Inference speed is tested on Snapdragon 855 ARM CPU for each SOTA model.

Model Name | Input Shape | mIou(%) | Speed(FPS)
---|---|---|---
PortraitNet | 224x224 | 95.20 |31.93
SINet | 224x224 | 93.76 | 78.12
PP-HumanSegV2 | 224x224 | 95.21 | 52.27
PP-HumanSegV2 | 192x192 | 94.87 | 81.96

Inference speed is tested on Arm CPU for model solutions.

Model Name | Input Shape | mIoU(%) | Inference Speed on Arm CPU(FPS)
---|---|---|---
PP-HumanSegV1 | 398x224 | 93.60 | 33.69
PP-HumanSegV2 | 398x224 | 97.50 | 35.23
PP-HumanSegV2 | 256x144 | 96.63 | 63.05

Inference speed is tested on Intel CPU for model solutions.

Model Name | Input Shape | mIoU(%) | Inference Speed on Intel CPU(FPS)
---|---|---|---
PP-HumanSegV1 | 398x224 | 93.60 | 36.02
PP-HumanSegV2 | 398x224 | 97.50 | 60.75
PP-HumanSegV2 | 256x144 | 96.63 | 70.67


## 2. Reference
Ref: [https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/contrib/PP-HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/contrib/PP-HumanSeg)
