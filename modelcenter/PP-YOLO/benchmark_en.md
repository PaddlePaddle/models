## Benchmark
### PP-YOLO
|          Model           | GPU number | images/GPU |  Backbone  | input shape | Box AP<sup>val</sup> | Box AP<sup>test</sup> | V100 FP32(FPS) | V100 TensorRT FP16(FPS) |
|:------------------------:|:-------:|:-------------:|:----------:| :-------:| :------------------: | :-------------------: | :------------: | :---------------------: |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     608     |         44.8         |         45.2          |      72.9      |          155.6          |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     512     |         43.9         |         44.4          |      89.9      |          188.4          |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     416     |         42.1         |         42.5          |      109.1      |          215.4          |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     320     |         38.9         |         39.3          |      132.2      |          242.2          |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     608     |         45.3         |         45.9          |      72.9      |          155.6          |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     512     |         44.4         |         45.0          |      89.9      |          188.4          |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     416     |         42.7         |         43.2          |      109.1      |          215.4          |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     320     |         39.5         |         40.1          |      132.2      |          242.2          |
| PP-YOLO               |     4      |     32     | ResNet18vd |     512     |         29.2         |         29.5          |      357.1      |          657.9          |
| PP-YOLO               |     4      |     32     | ResNet18vd |     416     |         28.6         |         28.9          |      409.8      |          719.4          |
| PP-YOLO               |     4      |     32     | ResNet18vd |     320     |         26.2         |         26.4          |      480.7      |          763.4          |

**Notes:**

- PP-YOLO is trained on COCO train2017 dataset and evaluated on val2017 & test-dev2017 dataset，Box AP<sup>test</sup> is evaluation results of `mAP(IoU=0.5:0.95)`.
- PP-YOLO used 8 GPUs for training and mini-batch size as 24 on each GPU, if GPU number and mini-batch size is changed, learning rate and iteration times should be adjusted according [FAQ](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/tutorials/FAQ).
- PP-YOLO inference speed is tesed on single Tesla V100 with batch size as 1, CUDA 10.2, CUDNN 7.5.1, TensorRT 5.1.2.2 in TensorRT mode.
- PP-YOLO FP32 inference speed testing uses inference model exported by `tools/export_model.py` and benchmarked by running `depoly/python/infer.py` with `--run_benchmark`. All testing results do not contains the time cost of data reading and post-processing(NMS), which is same as [YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet) in testing method.
- TensorRT FP16 inference speed testing exclude the time cost of bounding-box decoding(`yolo_box`) part comparing with FP32 testing above, which means that data reading, bounding-box decoding and post-processing(NMS) is excluded(test method same as [YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet) too)
- If you set `--run_benchmark=True`，you should install these dependencies at first, `pip install pynvml psutil GPUtil`.

### PP-YOLO for mobile
|    Model   | GPU number | images/GPU |  Model size  | input shape | Box AP<sup>val</sup> |  Box AP50<sup>val</sup> | Kirin 990 1xCore (FPS) |
|:------:|:-------:|:----------:|:--------:| :-------:| :------------------: |  :--------------------: | :--------------------: |
| PP-YOLO_MobileNetV3_large    |    4    |      32       |    28MB    |   320    |         23.2         |           42.6          |           14.1         |
| PP-YOLO_MobileNetV3_small    |    4    |      32       |    16MB    |   320    |         17.2         |           33.8          |           21.5         |
| PP-YOLO tiny                 |     8      |      32       |   4.2MB   |     320  |         20.6         |          92.3         |
| PP-YOLO tiny                 |     8      |      32       |   4.2MB   |     416  |         22.7         |          65.4         |

**Notes:**
- PP-YOLO for mobile is trained on COCO train2017 datast and evaluated on val2017 dataset，Box AP<sup>val</sup> is evaluation results of `mAP(IoU=0.5:0.95)`, Box AP<sup>val</sup> is evaluation results of `mAP(IoU=0.5)`.
- PP-YOLO-tiny used 8 GPUs for training and mini-batch size as 32 on each GPU. PP-YOLO_MobileNetV3 used 4 GPUs for training and mini-batch size as 32 on each GPU. If GPU number and mini-batch size is changed, learning rate and iteration times should be adjusted according [FAQ](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/tutorials/FAQ).
- PP-YOLO_MobileNetV3 inference speed is tested on Kirin 990 with 1 thread. PP-YOLO-tiny inference speed is tested on Kirin 990 with 4 threads by arm8.