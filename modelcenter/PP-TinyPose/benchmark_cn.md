# Keypoint Inference Benchmark

## Benchmark on Server
python部署测试详情如下表：

| Model | CPU + MKLDNN (thread=1) | CPU + MKLDNN (thread=4) | GPU | TensorRT (FP32) | TensorRT (FP16) |
| :------------------------ | :------: | :------: | :-----: | :---: | :---: |
| PP-TinyPose-128x96 | 25.2 ms | 14.1 ms | 2.7 ms | 0.9 ms | 0.8 ms |
| PP-TinyPose-256x192 | 82.4 ms | 36.1 ms | 3.0 ms | 1.5 ms | 1.1 ms |

**说明:**
- 以上测试均为python部署.
- 软硬件环境： NVIDIA T4 / PaddlePaddle(commit: 7df301f2fc0602745e40fa3a7c43ccedd41786ca) / CUDA10.1 / CUDNN7 / Python3.7 / TensorRT6.
- 测试样例： deploy/python/det_keypoint_unite_infer.py with image demo/000000014439.jpg. 输入的batch size为8。
- 时间仅包括推理时间 

c++部署测试详情如下表：

| Model | CPU + MKLDNN (thread=1) | CPU + MKLDNN (thread=4) | GPU | TensorRT (FP32) | TensorRT (FP16) |
| :------------------------ | :------: | :------: | :-----: | :---: | :---: |
| PP-TinyPose-128x96 | 24.06 ms | 13.05 ms | 2.43 ms | 0.75 ms | 0.72 ms |
| PP-TinyPose-256x192 | 82.73 ms | 36.25 ms | 2.57 ms | 1.38 ms | 1.15 ms |


**Notes:**
-  以上测试均为C++部署.
-  软硬件环境：NVIDIA T4 / PaddlePaddle(commit: 7df301f2fc0602745e40fa3a7c43ccedd41786ca) / CUDA10.1 / CUDNN7 / Python3.7 / TensorRT6.
- 测试样例： deploy/python/det_keypoint_unite_infer.py with image demo/000000014439.jpg. 输入的batch size为8。
- 时间仅包括推理时间 

## Benchmark on Mobile
我们在麒麟和高通骁龙设备上测试了baseline。详见下表。

| Model | Kirin 980 (1-thread) | Kirin 980 (4-threads)  | Qualcomm Snapdragon 845 (1-thread) | Qualcomm Snapdragon 845 (4-threads) | Qualcomm Snapdragon 660 (1-thread) | Qualcomm Snapdragon 660 (4-threads) |
| :------------------------ | :---: | :---: | :---: | :---: | :---: | :---: |
| PicoDet-s-192x192 (det) | 14.85 ms | 5.45 ms | 17.50 ms | 7.56 ms | 80.08 ms | 27.36 ms |
| PicoDet-s-320x320 (det) | 38.09 ms | 12.00 ms | 45.26 ms | 17.07 ms | 232.81 ms | 58.68 ms |
| PP-TinyPose-128x96 (pose) | 12.03 ms | 5.09 ms | 13.14 ms | 6.73 ms | 71.87 ms | 20.04 ms |

**Notes:**
- 以上测试基于Paddle Lite部署, 版本 v2.10-rc.
- 时间仅包括推理时间.
