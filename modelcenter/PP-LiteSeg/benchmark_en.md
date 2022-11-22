## 1. Inference Benchmark

### 1.1 Environment

* Segmentation accuracy (mIoU): PP-LightSeg is used for training and testing on Cityscapes dataset.
* Inference latency on Nvidia GPU: The hardware is Nvidia GPU (1080Ti). To keep the same as other methods, first use [infer_onnx_trt.py](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/deploy/python/infer_onnx_trt.py) to convert the model to ONNX format, and then use the native TRT prediction engine for testing.

### 1.2 Datasets

* Cityscapes dataset is used for testing.

### 1.3 Benchmark


<div align="center">

|Model|Encoder|Resolution|mIoU(Val)|mIoU(Test)|FPS|
|-|-|-|-|-|-|
ESPNet        | ESPNet      |  512x1024   | -    | 60.3 | 112.9 |
ESPNetV2      | ESPNetV2    |  512x1024   | 66.4 | 66.2 | -     |
SwiftNet      | ResNet18    |  1024x2048  | 75.4 | 75.5 | 39.9  |
BiSeNetV1     | Xception39  |  768x1536   | 69.0 | 68.4 | 105.8 |
BiSeNetV1-L   | ResNet18    |  768x1536   | 74.8 | 74.7 | 65.5  |
BiSeNetV2     | -           |  512x1024   | 73.4 | 72.6 | 156   |
BiSeNetV2-L   | -           |  512x1024   | 75.8 | 75.3 | 47.3  |
FasterSeg     | -           |  1024x2048  | 73.1 | 71.5 | 163.9 |
SFNet         | DF1         |  1024x2048  | -    | 74.5 | 121   |
STDC1-Seg50  | STDC1       |  512x1024   | 72.2 | 71.9 | 250.4 |
STDC2-Seg50  | STDC2       |  512x1024   | 74.2 | 73.4 | 188.6 |
STDC1-Seg75  | STDC1       |  768x1536   | 74.5 | 75.3 | 126.7 |
STDC2-Seg75  | STDC2       |  768x1536   | 77.0 | 76.8 | 97.0 |
PP-LiteSeg-T1 | STDC1      |  512x1024  | 73.1 | 72.0 | 273.6  |
PP-LiteSeg-B1 | STDC2      |  512x1024  | 75.3 | 73.9 | 195.3 |
PP-LiteSeg-T2 | STDC1      |  768x1536  | 76.0 | 74.9 | 143.6 |
PP-LiteSeg-B2 | STDC2      |  768x1536  | 78.2 | 77.5 | 102.6|

</div>

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/162148733-70be896a-eadb-4790-94e5-f48dad356b2d.png" width = "500" height = "430" alt="iou_fps"  />
</div>

## 2. Reference
Ref: https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/configs/pp_liteseg
