## 1. 训练Benchmark

### 1.1 软硬件环境

* PP-Matting训练过程中使用单卡GPU，batch size为4。

### 1.2 数据集

* 通用目标抠图数据集为Compositon-1k或Distinctions-646（使用该两者数据集需向作者进行申请），使用COCO2017和Pascal VOC 2012作为背景数据集。
* 人像抠图使用内部数据。

### 1.3 指标

|模型名称 | 模型简介 | 输入尺寸 |
|---|---|---|
|ppmatting_hrnet_w48 | 通用目标抠图 | 512 |
|ppmatting_hrnet_w18 | 人像抠图 | 512 |

## 2. 推理Benchmark

### 2.1 软硬件环境

* 模型推理速度测试采用单卡V100，batch size=1进行测试，使用CUDA 10.2, CUDNN 7.6.5, PaddlePaddle-gpu 2.3.2。

### 2.2 数据集

* 通用目标抠图：Compositon-1k或Distinctions-646中的测试集部分。
* 人像抠图：PPM-100和AIM-500中的人像部分，共195张, 记为PPM-AIM-195。

### 2.3 指标
| 模型 | 数据集 | SAD | MSE | Grad | Conn |Params(M) | FLOPs(G) | FPS |
| - | - | -| - | - | - | - | -| - |
| ppmatting_hrnet_w48 | Composition-1k | 46.22 | 0.005 | 22.69 | 45.40 | 86.3 | 165.4 | 24.4 |
| ppmatting_hrnet_w48 | Distinctions-646 | 40.69 | 0.009 | 43.91 |40.56 | 86.3 | 165.4 | 24.4 |
| ppmatting_hrnet_w18 | PPM-AIM-195 | 31.56|0.0022|31.80|30.13| 24.5 | 91.28 | 28.9 |

## 3. 相关使用说明
1. [https://github.com/PaddlePaddle/PaddleSeg/tree/develop/Matting](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/Matting)
