## 1. Training Benchmark

### 1.1 Environment

* The training process of PP-YOLO model uses 8 GPUs, every GPU batch size is 24 for training. If the number GPU and batch size of training do not use the above configuration, you should refer to the FAQ to adjust the learning rate and number of iterations.

* The training process of PP-YOLO_MobileNetV3 model uses 4GPU, every GPU batch size is 32 for training. If the number GPU and batch size for training do not use the above configuration, you should refer to the FAQ to adjust the learning rate and number of iterations.

* The training process of PP-YOLO-tiny model uses 8GPU, every GPU batch size is 32 for training. If the number of GPUs and batch size for training do not use the above configuration, you must refer to the FAQ to adjust the learning rate and number of iterations.

### 1.2 Datasets
The PP-YOLO model uses COCO dataset centralized train2017 as the training set and val2017 and test-dev2017 as the test set.

### 1.3 Benchmark

|模型名称 | 模型简介 |             模型体积 | 输入尺寸 | ips |
|---|---|---|---|---|
|ppyolov2_r50vd_dcn_1x_coco | 目标检测 |  | 640 |  |
|ppyolov2_r50vd_dcn_1x_coco | 目标检测 |  | 320 |  |


## 2. Inference Benchmark

### 2.1 Environment

* The PP-YOLO model's inference speed test is tested with single-card V100, batch size=1, CUDA 10.2, CUDNN 7.5.1, and TensorRT inference speed test using TensorRT 5.1.2.2.

* The PP-YOLO_MobileNetV3 model's inference speed test environment is configured as a Kirin 990 chip single-threaded.

* PP-YOLO-tiny model inference speed test environment is configured as Kirin 990 chip 4 threads, arm8 architecture.

### 2.2 Datasets
The PP-YOLO model uses COCO dataset centralized train2017 as the training set and val2017 and test-dev2017 as the test set.

### 2.3 Benchmark
PP-YOLOv2 (R50) mAP in the COCO test dataset rises from 45.9% to 49.5%, an increase of 3.6 percentage points compared to v1. FP32 FPS is up to 68.9FPS, FP16 FPS is up to 106.5FPS, surpassing YOLOv4 and even YOLOv5! If RestNet101 is used as the backbone network, PP-YOLOv2 (R101) has up to 50.3% mAP and 15.9% faster than YOLOv5x with the same accuracy!

![](https://raw.githubusercontent.com/PaddlePaddle/PaddleDetection/release/2.4/docs/images/ppyolo_map_fps.png)


## 3. Reference
Ref: https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/ppyolo/README_cn.md
