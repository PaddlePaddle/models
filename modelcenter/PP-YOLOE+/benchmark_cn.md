## Benchmark
|     模型      | Epoch | GPU个数 | 每GPU图片个数 |  骨干网络  | 输入尺寸 | Box AP<sup>val<br>0.5:0.95 | Box AP<sup>test<br>0.5:0.95 | Params(M) | FLOPs(G) | V100 FP32(FPS) | V100 TensorRT FP16(FPS) |
|:--------------:|:-----:|:-------:|:----------:|:----------:| :-------:|:--------------------------:|:---------------------------:|:---------:|:--------:|:---------------:| :---------------------: |
| PP-YOLOE+-s                  | 300 |     8      |    32    | cspresnet-s |     640     |       43.7        |        43.9         |   7.93    |  17.36   |   208.3   |  333.3   |
| PP-YOLOE+-m                  | 300 |     8      |    28    | cspresnet-m |     640     |       49. 8       |        50.0         |   23.43   |  49.91   |   123.4   |  208.3   |
| PP-YOLOE+-l                  | 300 |     8      |    20    | cspresnet-l |     640     |       52.9        |        53.3         |   52.20   |  110.07  |   78.1    |  149.2   |
| PP-YOLOE+-x                  | 300 |     8      |    16    | cspresnet-x |     640     |       54.7        |        54.9         |   98.42   |  206.59  |   45.0    |   95.2   |

**注意:**

- PP-YOLOE+模型使用COCO数据集中train2017作为训练集，使用val2017和test-dev2017作为测试集。
- PP-YOLOE+模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- PP-YOLOE+模型推理速度测试采用单卡V100，batch size=1进行测试，使用**CUDA 10.2**, **CUDNN 7.6.5**，TensorRT推理速度测试使用**TensorRT 6.0.1.8**。
- 参考[速度测试](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README_cn.md#%E9%80%9F%E5%BA%A6%E6%B5%8B%E8%AF%95)以复现PP-YOLOE+推理速度测试结果。
- 如果你设置了`--run_benchmark=True`, 你首先需要安装以下依赖`pip install pynvml psutil GPUtil`。