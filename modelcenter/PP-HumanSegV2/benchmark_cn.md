## 1. 推理 Benchmark

### 1.1 软硬件环境

* 测试肖像模型的精度mIoU：针对PP-HumanSeg-14k数据集，使用模型最佳输入尺寸进行测试，没有应用多尺度和flip等操作。
* 测试肖像模型在骁龙855 ARM CPU的推理耗时：基于PaddleLite预测库，小米9手机（骁龙855 CPU）、单线程、大核，使用模型最佳输入尺寸进行测试。
* 测试肖像模型在Intel CPU的推理耗时：基于Paddle Inference预测库，Intel(R) Xeon(R) Gold 6271C CPU。

### 1.2 数据集

* 使用PaddleSeg团队开源的PP-HumanSeg14k数据集进行测试。

### 1.3 指标

单个SOTA模型，在骁龙855 ARM CPU上进行测试速度。

模型 | 输入图像分辨率 | 精度mIoU(%) | 速度(FPS)
---|---|---|---
PortraitNet | 224x224 | 95.20 |31.93
SINet | 224x224 | 93.76 | 78.12
PP-HumanSegV2 | 224x224 | 95.21 | 52.27
PP-HumanSegV2 | 192x192 | 94.87 | 81.96

分割方案，在骁龙855 ARM CPU上进行测试速度。

模型 | 输入图像分辨率 | 精度mIoU(%) | ARM CPU速度(FPS)
---|---|---|---
PP-HumanSegV1 | 398x224 | 93.60 | 33.69
PP-HumanSegV2 | 398x224 | 97.50 | 35.23
PP-HumanSegV2 | 256x144 | 96.63 | 63.05

分割方案，在Intel CPU上进行测试速度。

模型 | 输入图像分辨率 | 精度mIoU(%) | Intel CPU速度(FPS)
---|---|---|---
PP-HumanSegV1 | 398x224 | 93.60 | 36.02
PP-HumanSegV2 | 398x224 | 97.50 | 60.75
PP-HumanSegV2 | 256x144 | 96.63 | 70.67


## 2. 相关使用说明
1. [https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/contrib/PP-HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/contrib/PP-HumanSeg)