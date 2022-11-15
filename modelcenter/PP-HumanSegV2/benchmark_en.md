## 1. Inference Benchmark

### 1.1 Environment

* Test the segmentation accuracy (mIoU): We test the above models on PP-HumanSeg-14K dataset with the best input shape.
* Test the inference time on Snapdragon 855 ARM CPU: Use PaddleLite, xiaomi9 (Snapdragon 855 CPU), single thread, the best input shape.
* Test the inference time on Intel CPU: Use Paddle Inference, Intel(R) Xeon(R) Gold 6271C CPU.

### 1.2 Datasets

* Use the PP-HumanSeg14k dataset opened by PaddleSeg team for testing.

### 1.3 Benchmark

Single SOTA model, test speed on Snapdragon 855 ARM CPU.

Model Name | Input Shape | mIou(%) | Speed(FPS)
---|---|---|---
PortraitNet | 224x224 | 95.20 |31.93
SINet | 224x224 | 93.76 | 78.12
PP-HumanSegV2 | 224x224 | 95.21 | 52.27
PP-HumanSegV2 | 192x192 | 94.87 | 81.96

For the segmentation scheme, test the speed on Snapdragon 855 ARM CPU.

Model Name | Input Shape | mIoU(%) | Inference Speed on Arm CPU(FPS)
---|---|---|---
PP-HumanSegV1 | 398x224 | 93.60 | 33.69
PP-HumanSegV2 | 398x224 | 97.50 | 35.23
PP-HumanSegV2 | 256x144 | 96.63 | 63.05

For the segmentation scheme, test the speed on Intel CPU.

Model Name | Input Shape | mIoU(%) | Inference Speed on Intel CPU(FPS)
---|---|---|---
PP-HumanSegV1 | 398x224 | 93.60 | 36.02
PP-HumanSegV2 | 398x224 | 97.50 | 60.75
PP-HumanSegV2 | 256x144 | 96.63 | 70.67


## 2. Reference
Ref: https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/contrib/PP-HumanSeg