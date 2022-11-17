## Benchmark
|     Model      | Epoch | GPU number | images/GPU |  backbone  | input shape | Box AP<sup>val<br>0.5:0.95 | Box AP<sup>test<br>0.5:0.95 | Params(M) | FLOPs(G) | V100 FP32(FPS) | V100 TensorRT FP16(FPS) |
|:--------------:|:-----:|:-------:|:----------:|:----------:| :-------:|:--------------------------:|:---------------------------:|:---------:|:--------:|:---------------:| :---------------------: |
| PP-YOLOE-s                  | 300 |     8      |    32    | cspresnet-s |     640     |       43.0        |        43.2         |   7.93    |  17.36   |   208.3   |  333.3   |
| PP-YOLOE-m                  | 300 |     8      |    28    | cspresnet-m |     640     |       49.0        |        49.1         |   23.43   |  49.91   |   123.4   |  208.3   |
| PP-YOLOE-l                  | 300 |     8      |    20    | cspresnet-l |     640     |       51.4        |        51.6         |   52.20   |  110.07  |   78.1    |  149.2   |
| PP-YOLOE-x                  | 300 |     8      |    16    | cspresnet-x |     640     |       52.3        |        52.4         |   98.42   |  206.59  |   45.0    |   95.2   |


**Notes:**

- PP-YOLOE is trained on COCO train2017 dataset and evaluated on val2017 & test-dev2017 dataset.
- PP-YOLOE used 8 GPUs for mixed precision training, if **GPU number** or **mini-batch size** is changed, **learning rate** should be adjusted according to the formula **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)**.
- PP-YOLOE inference speed is tesed on single Tesla V100 with batch size as 1, **CUDA 10.2**, **CUDNN 7.6.5**, **TensorRT 6.0.1.8** in TensorRT mode.
- Refer to [Speed testing](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README.md#speed-testing) to reproduce the speed testing results of PP-YOLOE.
- If you set `--run_benchmark=True`，you should install these dependencies at first, `pip install pynvml psutil GPUtil`.