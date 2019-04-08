PaddleCV
========

图像分类
--------

图像分类是根据图像的语义信息对不同类别图像进行区分，是计算机视觉中重要的基础问题，是物体检测、图像分割、物体跟踪、行为分析、人脸识别等其他高层视觉任务的基础，在许多领域都有着广泛的应用。如：安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

在深度学习时代，图像分类的准确率大幅度提升，在图像分类任务中，我们向大家介绍了如何在经典的数据集ImageNet上，训练常用的模型，包括AlexNet、VGG、GoogLeNet、ResNet、Inception-v4、MobileNet、DPN(Dual Path Network)、SE-ResNeXt模型，也开源了[训练的模型](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README_cn.md#已有模型及其性能) 方便用户下载使用。同时提供了能够将Caffe模型转换为PaddlePaddle
Fluid模型配置和参数文件的工具。

-  [AlexNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)
-  [VGG](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)
-  [GoogleNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)
-  [Residual Network](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)
-  [Inception-v4](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)
-  [MobileNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)
-  [Dual Path Network](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)
-  [SE-ResNeXt](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)
-  [Caffe模型转换为Paddle Fluid配置和模型文件工具](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/caffe2fluid)

目标检测
--------

目标检测任务的目标是给定一张图像或是一个视频帧，让计算机找出其中所有目标的位置，并给出每个目标的具体类别。对于人类来说，目标检测是一个非常简单的任务。然而，计算机能够“看到”的是图像被编码之后的数字，很难解图像或是视频帧中出现了人或是物体这样的高层语义概念，也就更加难以定位目标出现在图像中哪个区域。与此同时，由于目标会出现在图像或是视频帧中的任何位置，目标的形态千变万化，图像或是视频帧的背景千差万别，诸多因素都使得目标检测对计算机来说是一个具有挑战性的问题。

在目标检测任务中，我们介绍了如何基于[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 、[MS COCO](http://cocodataset.org/#home) 数据训练通用物体检测模型，当前介绍了SSD算法，SSD全称Single Shot MultiBox Detector，是目标检测领域较新且效果较好的检测算法之一，具有检测速度快且检测精度高的特点。

开放环境中的检测人脸，尤其是小的、模糊的和部分遮挡的人脸也是一个具有挑战的任务。我们也介绍了如何基于 [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace) 数据训练百度自研的人脸检测PyramidBox模型，该算法于2018年3月份在WIDER FACE的多项评测中均获得 [第一名](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)。

Faster RCNN模型是典型的两阶段目标检测器，相较于传统提取区域的方法，通过RPN网络共享卷积层参数大幅提高提取区域的效率，并提出高质量的候选区域。

Mask RCNN模型是基于Faster RCNN模型的经典实例分割模型，在原有Faster RCNN模型基础上添加分割分支，得到掩码结果，实现了掩码和类别预测关系的解藕。

-  [Single Shot MultiBox Detector](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/object_detection/README_cn.md)
-  [Face Detector: PyramidBox](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/face_detection/README_cn.md)
-  [Faster RCNN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/rcnn/README_cn.md)
-  [Mask RCNN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/rcnn/README_cn.md)

图像语义分割
------------

图像语意分割顾名思义是将图像像素按照表达的语义含义的不同进行分组/分割，图像语义是指对图像内容的理解，例如，能够描绘出什么物体在哪里做了什么事情等，分割是指对图片中的每个像素点进行标注，标注属于哪一类别。近年来用在无人车驾驶技术中分割街景来避让行人和车辆、医疗影像分析中辅助诊断等。

在图像语义分割任务中，我们介绍如何基于图像级联网络(Image Cascade
Network,ICNet)进行语义分割，相比其他分割算法，ICNet兼顾了准确率和速度。

-  [ICNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/icnet)

图像生成
-----------

图像生成是指根据输入向量，生成目标图像。这里的输入向量可以是随机的噪声或用户指定的条件向量。具体的应用场景有：手写体生成、人脸合成、风格迁移、图像修复等。当前的图像生成任务主要是借助生成对抗网络（GAN）来实现。
生成对抗网络（GAN）由两种子网络组成：生成器和识别器。生成器的输入是随机噪声或条件向量，输出是目标图像。识别器是一个分类器，输入是一张图像，输出是该图像是否是真实的图像。在训练过程中，生成器和识别器通过不断的相互博弈提升自己的能力。

在图像生成任务中，我们介绍了如何使用DCGAN和ConditioanlGAN来进行手写数字的生成，另外还介绍了用于风格迁移的CycleGAN.

- [DCGAN & ConditionalGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/gan/c_gan)
- [CycleGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/gan/cycle_gan)

场景文字识别
------------

许多场景图像中包含着丰富的文本信息，对理解图像信息有着重要作用，能够极大地帮助人们认知和理解场景图像的内容。场景文字识别是在图像背景复杂、分辨率低下、字体多样、分布随意等情况下，将图像信息转化为文字序列的过程，可认为是一种特别的翻译过程：将图像输入翻译为自然语言输出。场景图像文字识别技术的发展也促进了一些新型应用的产生，如通过自动识别路牌中的文字帮助街景应用获取更加准确的地址信息等。

在场景文字识别任务中，我们介绍如何将基于CNN的图像特征提取和基于RNN的序列翻译技术结合，免除人工定义特征，避免字符分割，使用自动学习到的图像特征，完成字符识别。当前，介绍了CRNN-CTC模型和基于注意力机制的序列到序列模型。

-  [CRNN-CTC模型](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/ocr_recognition)
-  [Attention模型](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/ocr_recognition)


度量学习
-------


度量学习也称作距离度量学习、相似度学习，通过学习对象之间的距离，度量学习能够用于分析对象时间的关联、比较关系，在实际问题中应用较为广泛，可应用于辅助分类、聚类问题，也广泛用于图像检索、人脸识别等领域。以往，针对不同的任务，需要选择合适的特征并手动构建距离函数，而度量学习可根据不同的任务来自主学习出针对特定任务的度量距离函数。度量学习和深度学习的结合，在人脸识别/验证、行人再识别(human Re-ID)、图像检索等领域均取得较好的性能，在这个任务中我们主要介绍了基于Fluid的深度度量学习模型，包含了三元组、四元组等损失函数。

- [Metric Learning](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning)


视频分类
-------

视频分类是视频理解任务的基础，与图像分类不同的是，分类的对象不再是静止的图像，而是一个由多帧图像构成的、包含语音数据、包含运动信息等的视频对象，因此理解视频需要获得更多的上下文信息，不仅要理解每帧图像是什么、包含什么，还需要结合不同帧，知道上下文的关联信息。视频分类方法主要包含基于卷积神经网络、基于循环神经网络、或将这两者结合的方法。该任务中我们介绍基于Fluid的视频分类模型，目前包含Temporal Segment Network(TSN)模型，后续会持续增加更多模型。


- [TSN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/video_classification)
