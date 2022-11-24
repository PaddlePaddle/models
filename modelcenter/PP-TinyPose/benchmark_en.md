# Keypoint Inference Benchmark

## Benchmark on Server
We tested benchmarks in different runtime environments。 See the table below for details.

| Model | CPU + MKLDNN (thread=1) | CPU + MKLDNN (thread=4) | GPU | TensorRT (FP32) | TensorRT (FP16) |
| :------------------------ | :------: | :------: | :-----: | :---: | :---: |
| PP-TinyPose-128x96 | 25.2 ms | 14.1 ms | 2.7 ms | 0.9 ms | 0.8 ms |
| PP-TinyPose-256x192 | 82.4 ms | 36.1 ms | 3.0 ms | 1.5 ms | 1.1 ms |

**Notes:**
- These tests above are based Python deployment.
- The environment is NVIDIA T4 / PaddlePaddle(commit: 7df301f2fc0602745e40fa3a7c43ccedd41786ca) / CUDA10.1 / CUDNN7 / Python3.7 / TensorRT6.
- The test is based on deploy/python/det_keypoint_unite_infer.py with image demo/000000014439.jpg. And input batch size for keypoint model is set to 8.
- The time only includes inference time.


| Model | CPU + MKLDNN (thread=1) | CPU + MKLDNN (thread=4) | GPU | TensorRT (FP32) | TensorRT (FP16) |
| :------------------------ | :------: | :------: | :-----: | :---: | :---: |
| PP-TinyPose-128x96 | 24.06 ms | 13.05 ms | 2.43 ms | 0.75 ms | 0.72 ms |
| PP-TinyPose-256x192 | 82.73 ms | 36.25 ms | 2.57 ms | 1.38 ms | 1.15 ms |


**Notes:**
- These tests above are based C++ deployment.
- The environment is NVIDIA T4 / PaddlePaddle(commit: 7df301f2fc0602745e40fa3a7c43ccedd41786ca) / CUDA10.1 / CUDNN7 / Python3.7 / TensorRT6.
- The test is based on deploy/python/det_keypoint_unite_infer.py with image demo/000000014439.jpg. And input batch size for keypoint model is set to 8.
- The time only includes inference time.

## Benchmark on Mobile
We tested benchmarks on Kirin and Qualcomm Snapdragon devices. See the table below for details.

| Model | Kirin 980 (1-thread) | Kirin 980 (4-threads)  | Qualcomm Snapdragon 845 (1-thread) | Qualcomm Snapdragon 845 (4-threads) | Qualcomm Snapdragon 660 (1-thread) | Qualcomm Snapdragon 660 (4-threads) |
| :------------------------ | :---: | :---: | :---: | :---: | :---: | :---: |
| PicoDet-s-192x192 (det) | 14.85 ms | 5.45 ms | 17.50 ms | 7.56 ms | 80.08 ms | 27.36 ms |
| PicoDet-s-320x320 (det) | 38.09 ms | 12.00 ms | 45.26 ms | 17.07 ms | 232.81 ms | 58.68 ms |
| PP-TinyPose-128x96 (pose) | 12.03 ms | 5.09 ms | 13.14 ms | 6.73 ms | 71.87 ms | 20.04 ms |

**Notes:**
- These tests above are based Paddle Lite deployment, and version is v2.10-rc.
- The time only includes inference time.
