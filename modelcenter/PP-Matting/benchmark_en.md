## 1. Training Benchmark

### 1.1 Environment

* The training process of PP-Matting model uses one GPU and batch size 4.

### 1.2 Datasets

* The common object matting dataset is Compositon-1k or Distinctins-646 (the use of both datasets are requested from the author). COCO2017 and Pascal VOC 2012 are used as the background datasets.
* Human matting uses the private dataset.

### 1.3 Benchmark
|Model | Description | Input shape |
|---|---|---|
|ppmatting_hrnet_w48 | Common object matting | 512 |
|ppmatting_hrnet_w18 | Human matting | 512 |

## 2. Inference Benchmark

### 2.1 Environment

* The PP-Matting model's inference speed test is tested with one V100, batch size=1, CUDA 10.2, CUDNN 7.6.5, PaddlePaddle-gpu 2.3.2.

### 2.2 Datasets
* Common object matting: the test dataset of Compositon-1k or Distinctions-646.
* Human matting: the human image of PPM-100 and AIM-500, total 195 images, named PPM-AIM-195.

### 2.3 Benchmark
| Model | Dataset | SAD | MSE | Grad | Conn |Params(M) | FLOPs(G) | FPS |
| - | - | -| - | - | - | - | -| - |
| ppmatting_hrnet_w48 | Composition-1k | 46.22 | 0.005 | 22.69 | 45.40 | 86.3 | 165.4 | 24.4 |
| ppmatting_hrnet_w48 | Distinctions-646 | 40.69 | 0.009 | 43.91 |40.56 | 86.3 | 165.4 | 24.4 |
| ppmatting_hrnet_w18 | PPM-AIM-195 | 31.56|0.0022|31.80|30.13| 24.5 | 91.28 | 28.9 |

## 3. Reference
1. [https://github.com/PaddlePaddle/PaddleSeg/tree/develop/Matting](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/Matting)
