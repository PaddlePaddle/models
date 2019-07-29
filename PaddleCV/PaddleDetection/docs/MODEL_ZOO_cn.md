# 模型库和基线

- Python 2.7.1
- PaddlePaddle 1.5
- CUDA 9.0
- CUDNN 7.4
- NCCL 2.1.2

## 通用设置

- SSD模型在VOC数据集中训练和测试，其余所有模型均在COCO17数据集中训练和测试。
- 除非特殊说明，所有ResNet骨干网络采用[ResNet-B](https://arxiv.org/pdf/1812.01187)结构。
- 对于RCNN和RetinaNet系列模型，训练阶段仅使用水平翻转作为数据增强，测试阶段不使用数据增强。

## 训练策略

- 我们采用和[Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#training-schedules)相同的训练策略。
- 1x 策略表示：在总batch size为16时，初始学习率为0.02，在6万轮和8万轮后学习率分别下降10倍，最终训练9万轮。在总batch size为8时，初始学习率为0.01，在12万轮和16万轮后学习率分别下降10倍，最终训练18万轮。
- 2x 策略为1x策略的两倍，同时学习率调整位置也为1x的两倍。

## ImageNet预训练模型

Paddle提供基于ImageNet的骨架网络预训练模型。所有预训练模型均通过标准的Imagenet-1k数据集训练得到。[下载链接](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#supported-models-and-performances)

- 注：ResNet50模型通过余弦学习率调整策略训练得到。[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar)

## 基线

### Faster & Mask R-CNN

| 骨架网络             | 网络类型           | 每张GPU图片个数 | 学习率策略 | Box AP | Mask AP |                           下载                          |
| :------------------- | :------------- | :-----: | :-----: | :----: | :-----: | :----------------------------------------------------------: |
| ResNet50             | Faster         |    1    |   1x    |  35.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar) |
| ResNet50             | Faster         |    1    |   2x    |  37.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_2x.tar) |
| ResNet50             | Mask           |    1    |   1x    |  36.5  |  32.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_1x.tar) |
| ResNet50             | Mask           |    1    |   2x    |  38.2  |  33.4   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_2x.tar) |
| ResNet50-vd          | Faster         |    1    |   1x    |  36.4  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_1x.tar) |
| ResNet50-FPN         | Faster         |    2    |   1x    |  37.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Faster         |    2    |   2x    |  37.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_2x.tar) |
| ResNet50-FPN         | Mask           |    1    |   1x    |  37.9  |  34.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Mask           |    1    |   2x    |  38.7  |  34.7   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_2x.tar) |
| ResNet50-FPN         | Cascade Faster |    2    |   1x    |  40.9  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_r50_fpn_1x.tar) |
| ResNet50-vd-FPN      | Faster         |    2    |   2x    |  38.9  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar) |
| ResNet50-vd-FPN      | Mask           |    1    |   2x    |  39.8  |  35.4   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar) |
| ResNet101            | Faster         |    1    |   1x    |  38.3  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_1x.tar) |
| ResNet101-FPN        | Faster         |    1    |   1x    |  38.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_1x.tar) |
| ResNet101-FPN        | Faster         |    1    |   2x    |  39.1  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_2x.tar) |
| ResNet101-FPN        | Mask           |    1    |   1x    |  39.5  |  35.2   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_fpn_1x.tar) |
| ResNet101-vd-FPN     | Faster         |    1    |   1x    |  40.5  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_1x.tar) |
| ResNet101-vd-FPN     | Faster         |    1    |   2x    |  40.8  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_2x.tar) |
| ResNet101-vd-FPN     | Mask           |    1    |   1x    |  41.4  |  36.8   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_vd_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   1x    |  42.2  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_x101_vd_64x4d_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Faster         |    1    |   2x    |  41.7  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_x101_vd_64x4d_fpn_2x.tar) |
| ResNeXt101-vd-FPN    | Mask           |    1    |   1x    |  42.9  |  37.9   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_x101_vd_64x4d_fpn_1x.tar) |
| ResNeXt101-vd-FPN    | Mask           |    1    |   2x    |  42.6  |  37.6   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_x101_vd_64x4d_fpn_2x.tar) |
| SENet154-vd-FPN      | Faster         |    1    |  1.44x  |  42.9  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_se154_vd_fpn_s1x.tar) |
| SENet154-vd-FPN      | Mask           |    1    |  1.44x  |  44.0  |  38.7   | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_se154_vd_fpn_s1x.tar) |

### Yolo v3

| 骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略 | Box AP | 下载 |
| :----------- | :--: | :-----: | :-----: | :----: | :-------: |
| DarkNet53    | 608  |    8    |   270e  |  38.9  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| DarkNet53    | 416  |    8    |   270e  |  37.5  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| DarkNet53    | 320  |    8    |   270e  |  34.8  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) |
| MobileNet-V1 | 608  |    8    |   270e  |  29.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNet-V1 | 416  |    8    |   270e  |  29.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| MobileNet-V1 | 320  |    8    |   270e  |  27.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
| ResNet34     | 608  |    8    |   270e  |  36.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet34     | 416  |    8    |   270e  |  34.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| ResNet34     | 320  |    8    |   270e  |  31.4  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |

### Yolo v3 基于Pasacl VOC数据集

| 骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略 | Box AP | 下载 |
| :----------- | :--: | :-----: | :-----: | :----: | :-------: |
| DarkNet53    | 608  |    8    |   270e  |  83.5  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| DarkNet53    | 416  |    8    |   270e  |  83.6  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| DarkNet53    | 320  |    8    |   270e  |  82.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet_voc.tar) |
| MobileNet-V1 | 608  |    8    |   270e  |  76.2  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNet-V1 | 416  |    8    |   270e  |  76.7  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| MobileNet-V1 | 320  |    8    |   270e  |  75.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
| ResNet34     | 608  |    8    |   270e  |  82.6  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |
| ResNet34     | 416  |    8    |   270e  |  81.9  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |
| ResNet34     | 320  |    8    |   270e  |  80.1  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |

**注意事项:** Yolo v3在8卡，总batch size为64下训练270轮。数据增强包括：mixup, 随机颜色失真，随机剪裁，随机扩张，随机插值法，随机翻转。Yolo v3在训练阶段对minibatch采用随机reshape，可以采用相同的模型测试不同尺寸图片，我们分别提供了尺寸为608/416/320大小的测试结果。

### RetinaNet

|   骨架网络   | 每张GPU图片个数 | 学习率策略 | Box AP | 下载  |
| :-----------  | :-----: | :-----: | :----: | :-------: |
| ResNet50-FPN  |    2    |   1x    |  36.0  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r50_fpn_1x.tar)  |
| ResNet101-FPN |    2    |   1x    |  37.3  | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/retinanet_r101_fpn_1x.tar) |

**注意事项:** RetinaNet系列模型中，在总batch size为16下情况下，初始学习率改为0.01。

### SSD on Pascal VOC

|  骨架网络     | 输入尺寸 | 每张GPU图片个数 | 学习率策略 | Box AP | 下载  |
| :----------- | :--: | :-----: | :-----: | :----: | :-------: |
| MobileNet v1 | 300  |    32   |   120e  |  73.13  | [model](https://paddlemodels.bj.bcebos.com/object_detection/ssd_mobilenet_v1_voc.tar) |

**注意事项:** SSD在2卡，总batch size为64下训练120轮。数据增强包括：随机颜色失真，随机剪裁，随机扩张，随机翻转。
