[English](README.md) | 简体中文

# PaddleDetection

PaddleDetection的目的是为工业界和学术界提供丰富、易用的目标检测模型。不仅性能优越、易于部署，同时能够灵活的满足算法研究的需求。

<div align="center">
  <img src="demo/output/000000570688.jpg" />
</div>


## 简介

特性：

- 易部署:

  PaddleDetection的模型中使用的核心算子均通过C++或CUDA实现，基于PaddlePaddle的高性能推理引擎支持跨平台的推理部署的能力。

- 高灵活度：

  PaddleDetection模块化设计解耦各个组件，模型结构以及数据预处理流程，均可通过修改配置文件轻松实现可定制化。

- 高性能：

  基于PaddlePaddle的高性能底层框架，在模型训练速度、显存占用上有一定的优势。例如，YOLOv3的训练速度快于其他框架，Mask-RCNN(ResNet50)可以在Tesla V100 16GB环境下以每个GPU 4张图片输入实现多卡训练。

支持的模型结构：

|                    | ResNet | ResNet-vd <sup>[1](#vd)</sup> | ResNeXt-vd | SENet | MobileNet | DarkNet | VGG |
|--------------------|:------:|------------------------------:|:----------:|:-----:|:---------:|:-------:|:---:|
| Faster R-CNN       | ✓      |                             ✓ | x          | ✓     | ✗         | ✗       | ✗   |
| Faster R-CNN + FPN | ✓      |                             ✓ | ✓          | ✓     | ✗         | ✗       | ✗   |
| Mask R-CNN         | ✓      |                             ✓ | x          | ✓     | ✗         | ✗       | ✗   |
| Mask R-CNN + FPN   | ✓      |                             ✓ | ✓          | ✓     | ✗         | ✗       | ✗   |
| Cascade R-CNN      | ✓      |                             ✓ | ✗          | ✗     | ✗         | ✗       | ✗   |
| RetinaNet          | ✓      |                             ✗ | ✗          | ✗     | ✗         | ✗       | ✗   |
| YOLOv3             | ✓      |                             ✗ | ✗          | ✗     | ✓         | ✓       | ✗   |
| SSD                | ✗      |                             ✗ | ✗          | ✗     | ✓         | ✗       | ✓   |

<a name="vd">[1]</a> [ResNet-vd](https://arxiv.org/pdf/1812.01187) 模型提供了较大的精度提高和较少的性能损失。

扩展特性：

- [x] **Synchronized Batch Norm**: 目前在YOLOv3中使用。
- [x] **Group Norm**
- [x] **Modulated Deformable Convolution**
- [x] **Deformable PSRoI Pooling**

**注意:** Synchronized batch normalization 只能在多GPU环境下使用，不能在CPU环境或者单GPU环境下使用。


# 使用教程

- [安装说明](docs/INSTALL_cn.md)
- [快速开始](docs/QUICK_STARTED_cn.md)
- [训练、评估及参数说明](docs/GETTING_STARTED_cn.md)
- [数据预处理及自定义数据集](docs/DATA_cn.md)
- [配置模块设计和介绍](docs/CONFIG_cn.md)
- [详细的配置信息和参数说明示例](docs/config_example/)
- [IPython Notebook demo](demo/mask_rcnn_demo.ipynb)
- [迁移学习教程](docs/TRANSFER_LEARNING_cn.md)

# 模型库

- [模型库](docs/MODEL_ZOO_cn.md)
- [人脸检测模型](configs/face_detection/README_cn.md)
- [行人检测和车辆检测预训练模型](contrib/README_cn.md)


### 模型压缩
- [量化训练压缩示例](slim/quantization)
- [剪枝压缩示例](slim/prune)

### 推理部署

- [C++推理部署](inference/README.md)

### Benchmark

- [推理Benchmark]()



## 版本更新

### 10/2019

- 增加人脸检测模型BlazeFace、Faceboxes。
- 丰富基于COCO的模型，精度高达51.9%。
- 增加基于Object365的模型CACascadeRCNN。
- 增加行人检测和车辆检测预训练模型。
- 增加跨平台的C++推理部署方案。
- 增加模型压缩示例。


### 2/9/2019
- 增加GroupNorm模型。
- 增加CascadeRCNN+Mask模型。

#### 5/8/2019
- 增加Modulated Deformable Convolution系列模型。

#### 7/22/2019

- 增加检测库中文文档
- 修复R-CNN系列模型训练同时进行评估的问题
- 新增ResNext101-vd + Mask R-CNN + FPN模型
- 新增基于VOC数据集的YOLOv3模型

#### 7/3/2019

- 首次发布PaddleDetection检测库和检测模型库
- 模型包括：Faster R-CNN, Mask R-CNN, Faster R-CNN+FPN, Mask
  R-CNN+FPN, Cascade-Faster-RCNN+FPN, RetinaNet, YOLOv3, 和SSD.

## 如何贡献代码

我们非常欢迎你可以为PaddleDetection提供代码，也十分感谢你的反馈。
