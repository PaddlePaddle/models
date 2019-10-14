# PaddleDetection 特色垂类检测模型

我们提供了针对不同场景的基于PaddlePaddle的检测模型，用户可以下载模型进行使用。

## 车辆检测（Vehicle Detection）

车辆检测的主要应用之一是交通监控。在这样的监控场景中，待检测的车辆多为道路红绿灯柱上的摄像头拍摄所得。

### 1. 模型结构

Backbone为Dacknet53的Yolo V3。

### 2. 精度指标

模型在我们业务数据上的精度指标为：

IOU=.50:.05:.95时的AP为0.545。
IOU=.5时的AP为0.764。

### 3. 预测

用户可以使用我们训练好的模型进行车辆检测：

```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python -u tools/infer.py -c contrib/VehicleDetection/vehicle_yolov3_darknet.yml \
                         -o weights= \ 
                         --infer_dir contrib/VehicleDetection/demo \
                         --draw_threshold 0.3 \
                         --output_dir contrib/VehicleDetection/demo/output

```

预测结果示例：

![](VehicleDetection/demo/output/001.jpeg)

![](VehicleDetection/demo/output/005.png)

## 行人检测（Pedestrian Detection）

行人检测的主要应用有智能监控。在监控场景中，大多是从公共区域的监控摄像头视角拍摄行人，获取图像后再进行行人检测。

### 1. 模型结构

Backbone为Dacknet53的Yolo V3

### 2. 精度指标

模型在针对监控场景的业务数据上精度指标为：

IOU=0.5  mAP: 0.792
IOU=0.5-0.95  mAP: 0.518

### 3. 预测

用户可以使用我们训练好的模型进行行人检测：

```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python -u tools/infer.py -c contrib/PedestrianDetection/pedestrian_yolov3_darknet.yml \ 
                         -o weights= \
                         --infer_dir contrib/PedestrianDetection/demo \ 
                         --draw_threshold 0.3 \
                         --output_dir contrib/PedestrianDetection/demo/output
```

预测结果示例：
![](PedestrianDetection/demo/output/001.png)
![](PedestrianDetection/demo/output/004.png)
