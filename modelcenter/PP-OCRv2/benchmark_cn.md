## 1. 训练Benchmark

### 1.1 软硬件环境

* PP-YOLO模型训练过程中使用8 GPUs，每GPU batch size为24进行训练，如训练GPU数和batch size不使用上述配置，须参考FAQ调整学习率和迭代次数。

* PP-YOLO_MobileNetV3 模型训练过程中使用4GPU，每GPU batch size为32进行训练，如训练GPU数和batch size不使用上述配置，须参考FAQ调整学习率和迭代次数。

* PP-YOLO-tiny 模型训练过程中使用8GPU，每GPU batch size为32进行训练，如训练GPU数和batch size不使用上述配置，须参考FAQ调整学习率和迭代次数。

### 1.2 数据集
PP-YOLO模型使用COCO数据集中train2017作为训练集，使用val2017和test-dev2017作为测试集.

### 1.3 指标 （字段可根据模型情况，自行定义）

|模型名称 | 模型简介 |             模型体积 | 输入尺寸 | ips |
|---|---|---|---|---|
|ppyolov2_r50vd_dcn_1x_coco | 目标检测 |  | 640 |  |
|ppyolov2_r50vd_dcn_1x_coco | 目标检测 |  | 320 |  |



## 2. 推理 Benchmark

### 2.1 软硬件环境

* PP-YOLO模型推理速度测试采用单卡V100，batch size=1进行测试，使用CUDA 10.2, CUDNN 7.5.1，TensorRT推理速度测试使用TensorRT 5.1.2.2。

* PP-YOLO_MobileNetV3 模型推理速度测试环境配置为麒麟990芯片单线程。

* PP-YOLO-tiny 模型推理速度测试环境配置为麒麟990芯片4线程，arm8架构。

### 2.2 数据集
PP-YOLO模型使用COCO数据集中train2017作为训练集，使用val2017和test-dev2017作为测试集.

### 2.3 指标（字段可根据模型情况，自行定义）

PP-YOLOv2（R50）在COCO test数据集mAP从45.9%达到了49.5%，相较v1提升了3.6个百分点，FP32 FPS高达68.9FPS，FP16 FPS高达106.5FPS，超越了YOLOv4甚至YOLOv5！如果使用RestNet101作为骨架网络，PP-YOLOv2（R101）的mAP更高达50.3%，并且比同等精度下的YOLOv5x快15.9%！




![](https://raw.githubusercontent.com/PaddlePaddle/PaddleDetection/release/2.4/docs/images/ppyolo_map_fps.png)

## 3. 相关使用说明
请参考：https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/ppyolo/README_cn.md
