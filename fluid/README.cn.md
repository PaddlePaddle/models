# models 简介



## 图像分类

图像分类是根据图像的语义信息对不同类别图像进行区分，是计算机视觉中重要的基础问题，是物体检测、图像分割、物体跟踪、行为分析、人脸识别等其他高层视觉任务的基础，在许多领域都有着广泛的应用。如：安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

在深度学习时代，图像分类的准确率大幅度提升，在图像分类任务中，我们向大家介绍了如何在经典的数据集ImageNet上，训练常用的模型，包括AlexNet、VGG、GoogLeNet、ResNet、Inception-v4、MobileNet、DPN(Dual Path Network)、SE-ResNeXt模型，也开源了[训练的模型](https://github.com/PaddlePaddle/models/blob/develop/fluid/image_classification/README_cn.md#已有模型及其性能)方便用户下载使用。同时提供了能够将Caffe模型转换为PaddlePaddle Fluid模型配置和参数文件的工具。

- [AlexNet](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models)
- [VGG](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models)
- [GoogleNet](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models)
- [Residual Network](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models)
- [Inception-v4](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models)
- [MobileNet](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models)
- [Dual Path Network](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models)
- [SE-ResNeXt](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models)
- [Caffe模型转换为Paddle Fluid配置和模型文件工具](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/caffe2fluid)

## 目标检测

目标检测任务的目标是给定一张图像或是一个视频帧，让计算机找出其中所有目标的位置，并给出每个目标的具体类别。对于人类来说，目标检测是一个非常简单的任务。然而，计算机能够“看到”的是图像被编码之后的数字，很难解图像或是视频帧中出现了人或是物体这样的高层语义概念，也就更加难以定位目标出现在图像中哪个区域。与此同时，由于目标会出现在图像或是视频帧中的任何位置，目标的形态千变万化，图像或是视频帧的背景千差万别，诸多因素都使得目标检测对计算机来说是一个具有挑战性的问题。

在目标检测任务中，我们介绍了如何基于[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)、[MS COCO](http://cocodataset.org/#home)数据的训练目标检测算法SSD，SSD全称Single Shot MultiBox Detector，是目标检测领域较新且效果较好的检测算法之一，具有检测速度快且检测精度高的特点，并开源了训练好的[MobileNet-SSD模型](https://github.com/PaddlePaddle/models/blob/develop/fluid/object_detection/README_cn.md#模型发布)。

- [Single Shot MultiBox Detector](https://github.com/PaddlePaddle/models/blob/develop/fluid/object_detection/README_cn.md)


## 图像语义分割

图像语意分割顾名思义是将图像像素按照表达的语义含义的不同进行分组/分割，图像语义是指对图像内容的理解，例如，能够描绘出什么物体在哪里做了什么事情等，分割是指对图片中的每个像素点进行标注，标注属于哪一类别。近年来用在无人车驾驶技术中分割街景来避让行人和车辆、医疗影像分析中辅助诊断等。

在图像语义分割任务中，我们介绍如何基于图像级联网络(Image Cascade Network,ICNet)进行语义分割，相比其他分割算法，ICNet兼顾了准确率和速度。


- [ICNet](https://github.com/PaddlePaddle/models/tree/develop/fluid/icnet)


## 场景文字识别

许多场景图像中包含着丰富的文本信息，对理解图像信息有着重要作用，能够极大地帮助人们认知和理解场景图像的内容。场景文字识别是在图像背景复杂、分辨率低下、字体多样、分布随意等情况下，将图像信息转化为文字序列的过程，可认为是一种特别的翻译过程：将图像输入翻译为自然语言输出。场景图像文字识别技术的发展也促进了一些新型应用的产生，如通过自动识别路牌中的文字帮助街景应用获取更加准确的地址信息等。

在场景文字识别任务中，我们介绍如何将基于CNN的图像特征提取和基于RNN的序列翻译技术结合，免除人工定义特征，避免字符分割，使用自动学习到的图像特征，完成端到端地无约束字符定位和识别。当前，介绍了CRNN-CTC模型，后续会引入基于注意力机制的序列到序列模型。

- [CRNN-CTC模型](https://github.com/PaddlePaddle/models/tree/develop/fluid/ocr_recognition)
