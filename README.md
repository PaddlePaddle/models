# PaddlePaddle Models

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/models) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle 提供了丰富的计算单元，使得用户可以采用模块化的方法解决各种学习问题。在此Repo中，我们展示了如何用 PaddlePaddle来解决常见的机器学习任务，提供若干种不同的易学易用的神经网络模型。PaddlePaddle用户可领取**免费Tesla V100在线算力资源**，高效训练模型，**每日登陆即送12小时**，**连续五天运行再加送48小时**，[前往使用免费算力](http://ai.baidu.com/support/news?action=detail&id=981)。

## 目录
* [智能视觉(PaddleCV)](#PaddleCV)
  * [图像分类](#图像分类)
  * [目标检测](#目标检测)
  * [图像分割](#图像分割)
  * [关键点检测](#关键点检测)
  * [图像生成](#图像生成)
  * [场景文字识别](#场景文字识别)
  * [度量学习](#度量学习)
  * [视频](#视频)
* [智能文本处理(PaddleNLP)](#PaddleNLP)
  * [NLP 基础技术](#NLP-基础技术)
  * [NLP 核心技术](#NLP-核心技术)
  * [NLP 系统应用](#NLP-系统应用)
* [智能推荐(PaddleRec)](#PaddleRec)
* [智能语音(PaddleSpeech)](#PaddleSpeech)
* [快速下载模型库](#快速下载模型库)

## PaddleCV

### 图像分类

[图像分类](https://github.com/PaddlePaddle/PaddleClas) 是根据图像的语义信息对不同类别图像进行区分，是计算机视觉中重要的基础问题，是物体检测、图像分割、物体跟踪、行为分析、人脸识别等其他高层视觉任务的基础，在许多领域都有着广泛的应用。如：安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

| **模型名称** | **模型简介** | **数据集** | **评估指标 top-1/top-5 accuracy** |
| - | - | - | - |
| [AlexNet](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 首次在 CNN 中成功的应用了 ReLU, Dropout 和 LRN，并使用 GPU 进行运算加速 | ImageNet-2012验证集 | 56.72%/79.17% |
| [VGG19](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 在 AlexNet 的基础上使用 3*3 小卷积核，增加网络深度，具有很好的泛化能力 | ImageNet-2012验证集 | 72.56%/90.93% |
| [GoogLeNet](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 在不增加计算负载的前提下增加了网络的深度和宽度，性能更加优越 | ImageNet-2012验证集 | 70.70%/89.66% |
| [ResNet50](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | Residual Network，引入了新的残差结构，解决了随着网络加深，准确率下降的问题 | ImageNet-2012验证集 | 76.50%/93.00% |
| [ResNet200_vd](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 融合多种对 ResNet 改进策略，ResNet200_vd 的 top1 准确率达到 80.93% | ImageNet-2012验证集 | 80.93%/95.33% |
| [Inceptionv4](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 将 Inception 模块与 Residual Connection 进行结合，通过ResNet的结构极大地加速训练并获得性能的提升 | ImageNet-2012验证集 | 80.77%/95.26% |
| [MobileNetV1](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 将传统的卷积结构改造成两层卷积结构的网络，在基本不影响准确率的前提下大大减少计算时间，更适合移动端和嵌入式视觉应用 | ImageNet-2012验证集 | 70.99%/89.68% |
| [MobileNetV2](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | MobileNet结构的微调，直接在 thinner 的 bottleneck层上进行 skip learning 连接以及对 bottleneck layer 不进行 ReLu 非线性处理可取得更好的结果 | ImageNet-2012验证集 | 72.15%/90.65% |
| [SENet154_vd](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 在ResNeXt 基础、上加入了 SE(Sequeeze-and-Excitation) 模块，提高了识别准确率，在 ILSVRC 2017 的分类项目中取得了第一名 | ImageNet-2012验证集 | 81.40%/95.48% |
| [ShuffleNetV2](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | ECCV2018，轻量级 CNN 网络，在速度和准确度之间做了很好地平衡。在同等复杂度下，比 ShuffleNet 和 MobileNetv2 更准确，更适合移动端以及无人车领域 | ImageNet-2012验证集 | 70.03%/89.17% |
| [efficientNet](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 同时对模型的分辨率，通道数和深度。进行缩放，用极少的参数就可以达到SOTA的精度。 | ImageNet-2012验证集 | 77.38%/93.31% |
| [xception71](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 对inception-v3的改进，用深度可分离卷积代替普通卷积，降低参数量的同时提高了精度。 | ImageNet-2012验证集 | 81.11%/95.45% |
| [dpn107](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 融合了densenet和resnext的特点。 | ImageNet-2012验证集 | 80.89%/95.32% |
| [mobilenetV3_small_x1_0](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 在v2的基础上增加了se模块，并且使用hard-swish激活函数。在分类、检测、分割等视觉任务上都有不错表现。 | ImageNet-2012验证集 | 67.46%/87.12% |
| [DarkNet53](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 检测框架yolov3使用的backbone，在分类和检测任务上都有不错表现。 | ImageNet-2012验证集 | 78.04%/94.05% |
| [DenseNet161](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 提出了密集连接的网络结构，更加有利于信息流的传递。 | ImageNet-2012验证集 | 78.57%/94.14% |
| [ResNeXt152_vd_64x4d](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 提出了cardinatity的概念，用于作为模型复杂度的另外一个度量，并依据该概念有效地提升了模型精度。 | ImageNet-2012验证集 | 81.08%/95.34% |
| [SqueezeNet1_1](https://paddleclas.readthedocs.io/zh_CN/latest/models/index.html) | 提出了新的网络架构Fire Module，通过减少参数来进行模型压缩。 | ImageNet-2012验证集 | 60.08%/81.85% |

更多图像分类模型请参考 [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)。

### 目标检测

目标检测任务的目标是给定一张图像或是一个视频帧，让计算机找出其中所有目标的位置，并给出每个目标的具体类别。对于计算机而言，能够“看到”的是图像被编码之后的数字，但很难解图像或是视频帧中出现了人或是物体这样的高层语义概念，也就更加难以定位目标出现在图像中哪个区域。目标检测模型请参考 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)。

| 模型名称                                                     | 模型简介                                                     | 数据集     | 评估指标   mAP                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- | ------------------------------------------------------- |
| [SSD](https://github.com/PaddlePaddle/PaddleDetection) | 很好的继承了 MobileNet 预测速度快，易于部署的特点，能够很好的在多种设备上完成图像目标检测任务 | VOC07 test | mAP   = 73.32%                                          |
| [Faster-RCNN](https://github.com/PaddlePaddle/PaddleDetection) | 创造性地采用卷积网络自行产生建议框，并且和目标检测网络共享卷积网络，建议框数目减少，质量提高 | MS-COCO    | 基于ResNet 50  mAP(0.50: 0.95) = 36.7%                  |
| [Mask-RCNN](https://github.com/PaddlePaddle/PaddleDetection) | 经典的两阶段框架，在 Faster R-CNN模型基础上添加分割分支，得到掩码结果，实现了掩码和类别预测关系的解藕，可得到像素级别的检测结果。 | MS-COCO    | 基于ResNet 50   Mask mAP（0.50: 0.95） = 31.4%          |
| [RetinaNet](https://github.com/PaddlePaddle/PaddleDetection) | 经典的一阶段框架，由主干网络、FPN结构、和两个分别用于回归物体位置和预测物体类别的子网络组成。在训练过程中使用 Focal Loss，解决了传统一阶段检测器存在前景背景类别不平衡的问题，进一步提高了一阶段检测器的精度。 | MS-COCO    | 基于ResNet 50 mAP (0.50: 0.95) = 36%                    |
| [YOLOv3](https://github.com/PaddlePaddle/PaddleDetection) | 速度和精度均衡的目标检测网络，相比于原作者 darknet 中的 YOLO v3 实现，PaddlePaddle 实现参考了论文 [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf) 增加了 mixup，label_smooth 等处理，精度 (mAP(0.50: 0.95)) 相比于原作者提高了 4.7 个绝对百分点，在此基础上加入 synchronize batch normalization, 最终精度相比原作者提高 5.9 个绝对百分点。 | MS-COCO    | 基于DarkNet   mAP(0.50: 0.95)=   38.9%                  |
| [PyramidBox](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/face_detection) | **PyramidBox** **模型是百度自主研发的人脸检测模型**，利用上下文信息解决困难人脸的检测问题，网络表达能力高，鲁棒性强。于18年3月份在 WIDER Face 数据集上取得第一名 | WIDER FACE | mAP   （Easy/Medium/Hard   set）=   96.0%/ 94.8%/ 88.8% |
| [Cascade RCNN](https://github.com/PaddlePaddle/PaddleDetection) | Cascade R-CNN 在 Faster R-CNN 框架下，通过级联多个检测器，在训练过程中选取不同的 IoU 阈值，逐步提高目标定位的精度，从而获取优异的检测性能。 | MS-COCO    | 基于ResNet 50 mAP (0.50: 0.95) = 40.9%                  |
| [Faceboxes](https://github.com/PaddlePaddle/PaddleDetection) | 经典的人脸检测网络，被称为“高精度 CPU 实时人脸检测器”。网络中使用率 CReLU、density_prior_bo x等组件，使得模型的精度和速度得到平衡与提升。相比于 PyramidBox，预测与计算更快，模型更小，精度也保持高水平。 | WIDER FACE | mAP (Easy/Medium/Hard Set) = 0.898/0.872/0.752          |
| [BlazeFace](https://github.com/PaddlePaddle/PaddleDetection) | 高速的人脸检测网络，由5个单的和6个双 BlazeBlocks、和 SSD 的架构构成。它轻巧但性能良好，并且专为移动 GPU 推理量身定制。 | WIDER FACE | mAP Easy/Medium/Hard Set = 0.915/0.892/0.797            |

### 图像分割

图像语义分割顾名思义是将图像像素按照表达的语义含义的不同进行分组/分割，图像语义是指对图像内容的理解，例如，能够描绘出什么物体在哪里做了什么事情等，分割是指对图片中的每个像素点进行标注，标注属于哪一类别。近年来用在无人车驾驶技术中分割街景来避让行人和车辆、医疗影像分析中辅助诊断等。
图像语义分割模型请参考语义分割库[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)

| 模型名称                                                     | 模型简介                                                     | 数据集    | 评估指标        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------- | --------------- |
| [U-Net](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.4.0/turtorial/finetune_unet.md) |起源于医疗图像分割，整个网络是标准的encoder-decoder网络，特点是参数少，计算快，应用性强，对于一般场景适应度很高。U-Net最早于2015年提出，并在ISBI 2015 Cell Tracking Challenge取得了第一。经过发展，目前有多个变形和应用。| -- | -- |
| [ICNet](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.4.0/turtorial/finetune_icnet.md) | 主要用于图像实时语义分割，能够兼顾速度和准确性，易于线上部署。 | Cityscapes | Mean IoU=68.31%  |
| [PSPNet](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.4.0/turtorial/finetune_pspnet.md) | 起源于场景解析(Scene Parsing)领域。通过特殊设计的全局均值池化操作（global average pooling）和特征融合构造金字塔池化模块 (Pyramid Pooling Module)，来融合图像中不同区域的上下文信息。 | Cityscapes | Mean IoU=77.34% |
| [DeepLabv3+](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.4.0/turtorial/finetune_deeplabv3plus.md) | 通过 encoder-decoder 进行多尺度信息的融合，同时保留了原来的空洞卷积和 ASSP 层，其骨干网络使用了 Xception 模型，提高了语义分割的健壮性和运行速率。 | Cityscapes | Mean IoU=79.30% |
| [HRNet](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.4.0/turtorial/finetune_hrnet.md) | 在整个训练过程中始终维持高分辨率表示。 通过两个特性学习到更丰富的语义信息和细节信息：（1）从高分辨率到低分辨率并行连接各子网络，（2）反复交换跨分辨率子网络信息。 在人体姿态估计、语义分割和目标检测领域都取得了显著的性能提升。 | Cityscapes | Mean IoU=79.36% |
| [Fast-SCNN](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.4.0/turtorial/finetune_fast_scnn.md) | 一个面向实时的语义分割网络。在双分支的结构基础上，大量使用了深度可分离卷积和逆残差（inverted-residual）模块，并且使用特征融合构造金字塔池化模块 (Pyramid Pooling Module)来融合上下文信息。这使得Fast-SCNN在保持高效的情况下能学习到丰富的细节信息。 | Cityscapes | Mean IoU=69.64% |
| [PSPNet (res101)](https://github.com/PaddlePaddle/Research/tree/master/CV/SemSegPaddle) | 通过利用不同子区域和全局的上下文信息来增强语义分割质量，同时提出deeply supervised 的辅助loss去改善模型的优化 | Cityscapes | Mean IoU=78.1% |
| [GloRe (res101)](https://github.com/PaddlePaddle/Research/tree/master/CV/SemSegPaddle) | 提出一个轻量级的、可端到端训练的全局推理单元GloRe来高效推理image regions之间的关系，增强了模型上下文建模能力| Cityscapes | Mean IoU=78.4% |
| [PSPNet (res101)](https://github.com/PaddlePaddle/Research/tree/master/CV/SemSegPaddle) | -| PASCAL Context | Mean IoU=48.9  |
| [GloRe (res101)](https://github.com/PaddlePaddle/Research/tree/master/CV/SemSegPaddle) | -| PASCAL Context | Mean IoU=48.4 |

### 关键点检测

人体骨骼关键点检测 (Pose Estimation) 主要检测人体的一些关键点，如关节，五官等，通过关键点描述人体骨骼信息。人体骨骼关键点检测对于描述人体姿态，预测人体行为至关重要。是诸多计算机视觉任务的基础，例如动作分类，异常行为检测，以及自动驾驶等等。

| 模型名称                                                     | 模型简介                                                     | 数据集       | 评估指标     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ | ------------ |
| [Simple   Baselines](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/human_pose_estimation) | coco2018 关键点检测项目亚军方案，网络结构非常简单，效果达到 state of the art | COCO val2017 | AP =   72.7% |

### 图像生成

图像生成是指根据输入向量，生成目标图像。这里的输入向量可以是随机的噪声或用户指定的条件向量。具体的应用场景有：手写体生成、人脸合成、风格迁移、图像修复等。[gan](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan) 包含和图像生成相关的多个模型。

| 模型名称                                                     | 模型简介                                                     | 数据集     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- |
| [CGAN](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan) | 条件生成对抗网络，一种带条件约束的 GAN，使用额外信息对模型增加条件，可以指导数据生成过程 | Mnist      |
| [DCGAN](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan) | 深度卷积生成对抗网络，将 GAN 和卷积网络结合起来，以解决 GAN 训练不稳定的问题 | Mnist      |
| [Pix2Pix](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan) | 图像翻译，通过成对图片将某一类图片转换成另外一类图片，可用于风格迁移 | Cityscapes |
| [CycleGAN](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan) | 图像翻译，可以通过非成对的图片将某一类图片转换成另外一类图片，可用于风格迁移 | Cityscapes |
| [StarGAN](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan) | 多领域属性迁移，引入辅助分类帮助单个判别器判断多个属性，可用于人脸属性转换 | Celeba     |
| [AttGAN](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan) | 利用分类损失和重构损失来保证改变特定的属性，可用于人脸特定属性转换 | Celeba     |
| [STGAN](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan) | 人脸特定属性转换，只输入有变化的标签，引入 GRU 结构，更好的选择变化的属性 | Celeba     |
| [SPADE](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan) | 提出一种考虑空间语义信息的归一化方法，从而更好的保留语义信息，生成更为逼真的图像，可用于图像翻译。 | Cityscapes |

### 场景文字识别

场景文字识别是在图像背景复杂、分辨率低下、字体多样、分布随意等情况下，将图像信息转化为文字序列的过程，可认为是一种特别的翻译过程：将图像输入翻译为自然语言输出。

| 模型名称                                                     | 模型简介                                                     | 数据集                     | 评估指标       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------- | -------------- |
| [CRNN-CTC](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/ocr_recognition) | 使用 CTC model 识别图片中单行英文字符，用于端到端的文本行图片识别方法 | 单行不定长的英文字符串图片 | 错误率= 22.3%  |
| [OCR   Attention](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/ocr_recognition) | 使用 attention 识别图片中单行英文字符，用于端到端的自然场景文本识别 | 单行不定长的英文字符串图片 | 错误率 = 15.8% |

### 度量学习

度量学习也称作距离度量学习、相似度学习，通过学习对象之间的距离，度量学习能够用于分析对象时间的关联、比较关系，在实际问题中应用较为广泛，可应用于辅助分类、聚类问题，也广泛用于图像检索、人脸识别等领域。

| 模型名称                                                     | 模型简介                                                     | 数据集                         | 评估指标 Recall@Rank-1（使用arcmargin训练） |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------ | ------------------------------------------- |
| [ResNet50未微调](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/metric_learning) | 使用 arcmargin loss 训练的特征模型                           | Stanford   Online Product(SOP) | 78.11%                                      |
| [ResNet50使用triplet微调](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/metric_learning) | 在 arcmargin loss 基础上，使用 triplet loss 微调的特征模型   | Stanford   Online Product(SOP) | 79.21%                                      |
| [ResNet50使用quadruplet微调](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/metric_learning) | 在 arcmargin loss 基础上，使用 quadruplet loss 微调的特征模型 | Stanford   Online Product(SOP) | 79.59%                                      |
| [ResNet50使用eml微调](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/metric_learning) | 在 arcmargin loss 基础上，使用 eml loss 微调的特征模型       | Stanford   Online Product(SOP) | 80.11%                                      |
| [ResNet50使用npairs微调](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/metric_learning) | 在 arcmargin loss基础上，使用npairs loss 微调的特征模型      | Stanford   Online Product(SOP) | 79.81%                                      |


### 视频

PaddleCV全面开源了视频分类、动作定位 和 目标跟踪等视频任务的领先实用算法。视频数据包含语音、图像等多种信息，因此理解视频任务不仅需要处理语音和图像，还需要提取视频帧时间序列中的上下文信息。
视频分类模型提供了提取全局时序特征的方法，主要方式有卷积神经网络 (C3D, I3D, C2D等)，神经网络和传统图像算法结合 (VLAD 等)，循环神经网络等建模方法。
视频动作定位模型需要同时识别视频动作的类别和起止时间点，通常采用类似于图像目标检测中的算法在时间维度上进行建模。
视频摘要生成模型是对视频画面信息进行提取，并产生一段文字描述。视频查找模型则是基于一段文字描述，查找到视频中对应场景片段的起止时间点。这两类模型需要同时对视频图像和文本信息进行建模。
目标跟踪任务是在给定某视频序列中找到目标物体，并将不同帧中的物体一一对应，然后给出不同物体的运动轨迹，目标跟踪的主要应用在视频监控、人机交互等系统中。跟踪又分为单目标跟踪和多目标跟踪，当前在飞桨模型库中增加了单目标跟踪的算法。主要包括Siam系列算法和ATOM算法。

| 模型名称                                                     | 模型简介                                                     | 数据集                     | 评估指标    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------- | ----------- |
| [TSN](./PaddleCV/video) | ECCV'16 提出的基于 2D-CNN 经典解决方案 | Kinetics-400               | Top-1 = 67% |
| [Non-Local](./PaddleCV/video) | 视频非局部关联建模模型 | Kinetics-400               | Top-1 = 74% |
| [StNet](./PaddleCV/video) | AAAI'19 提出的视频联合时空建模方法 | Kinetics-400               | Top-1 = 69% |
| [TSM](./PaddleCV/video) | 基于时序移位的简单高效视频时空建模方法 | Kinetics-400               | Top-1 = 70% |
| [Attention   LSTM](./PaddleCV/video) | 常用模型，速度快精度高 | Youtube-8M                 | GAP   = 86% |
| [Attention   Cluster](./PaddleCV/video) | CVPR'18 提出的视频多模态特征注意力聚簇融合方法 | Youtube-8M                 | GAP   = 84% |
| [NeXtVlad](./PaddleCV/video) | 2nd-Youtube-8M 比赛第 3 名的模型 | Youtube-8M                 | GAP   = 87% |
| [C-TCN](./PaddleCV/video) | 2018 年 ActivityNet 夺冠方案 | ActivityNet1.3 | MAP=31%    |
| [BSN](./PaddleCV/video) | 为视频动作定位问题提供高效的 proposal 生成方法 | ActivityNet1.3 | AUC=66.64%    |
| [BMN](./PaddleCV/video) | 2019 年 ActivityNet 夺冠方案 | ActivityNet1.3 | AUC=67.19%    |
| [ETS](./PaddleCV/video) | 视频摘要生成领域的基准模型 | ActivityNet Captions | METEOR：10.0 |
| [TALL](./PaddleCV/video) | 视频Grounding方向的BaseLine模型 | TACoS | R1@IOU5=0.13 |
| [SiamFC](./PaddleCV/tracking) | ECCV’16提出的全卷积神经网络视频跟踪模型 | VOT2018 | EAO = 0.211 |
| [ATOM](./PaddleCV/tracking) | CVPR’19提出的两阶段目标跟踪模型 | VOT2018 | EAO = 0.399 |



### 3D视觉

计算机3D视觉技术是解决包含高度、宽度、深度三个方向信息的三维立体图像的分类、分割、检测、识别等任务的计算机技术，广泛地应用于如机器人、无人车、AR等领域。3D点云是3D图像数据的主要表达形式之一，基于3D点云的形状分类、语义分割、目标检测模型是3D视觉方向的基础任务。当前飞桨模型库开源了基于3D点云数据的用于分类、分割的PointNet++模型和用于检测的PointRCNN模型。

| 模型名称                                                     | 模型简介                                                     | 数据集                     | 评估指标    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------- | ----------- |
| [PointNet++](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/3d_vision/PointNet++) | 改进的PointNet网络，加入局部特征提取提高模型泛化能力 | ModelNet40(分类) / Indoor3D(分割) | 分类：Top-1 = 90% / 分割：Top-1 = 86% |
| [PointRCNN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/3d_vision/PointRCNN) | 自下而上的3D检测框生成方法 | KITTI(Car) | 3D AP@70(easy/median/hard) = 86.66/76.65/75.90 |

## PaddleNLP

[**PaddleNLP**](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP) 是基于 PaddlePaddle 深度学习框架开发的自然语言处理 (NLP) 工具，算法，模型和数据的开源项目。百度在 NLP 领域十几年的深厚积淀为 PaddleNLP 提供了强大的核心动力。使用 PaddleNLP，您可以得到：

- **丰富而全面的 NLP 任务支持：**
  - PaddleNLP 为您提供了多粒度，多场景的应用支持。涵盖了从[分词](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/lexical_analysis)，[词性标注](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/lexical_analysis)，[命名实体识别](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/lexical_analysis)等 NLP 基础技术，到[文本分类](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/sentiment_classification)，[文本相似度计算](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/similarity_net)，[语义表示](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/pretrain_language_models)，[文本生成](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/seq2seq)等 NLP 核心技术。同时，PaddleNLP 还提供了针对常见 NLP 大型应用系统（如[阅读理解](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/machine_reading_comprehension)，[对话系统](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/dialogue_system)，[机器翻译系统](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/machine_translation)等）的特定核心技术和工具组件，模型和预训练参数等，让您在 NLP 领域畅通无阻。
- **稳定可靠的 NLP 模型和强大的预训练参数：**
  - PaddleNLP集成了百度内部广泛使用的 NLP 工具模型，为您提供了稳定可靠的 NLP 算法解决方案。基于百亿级数据的预训练参数和丰富的预训练模型，助您轻松提高模型效果，为您的 NLP 业务注入强大动力。
- **持续改进和技术支持，零基础搭建 NLP 应用：**
  - PaddleNLP 为您提供持续的技术支持和模型算法更新，为您的 NLP 业务保驾护航。

### NLP 基础技术

| 任务类型     | 目录                                                         | 简介                                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 中文词法分析 | [LAC(Lexical Analysis of Chinese)](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/lexical_analysis) | 百度自主研发中文特色模型词法分析任务，集成了中文分词、词性标注和命名实体识别任务。输入是一个字符串，而输出是句子中的词边界和词性、实体类别。 |
| 词向量       | [Word2vec](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleRec/word2vec) | 提供单机多卡，多机等分布式训练中文词向量能力，支持主流词向量模型（skip-gram，cbow等），可以快速使用自定义数据训练词向量模型。 |
| 语言模型     | [Language_model](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/language_model) | 给定一个输入词序列（中文需要先分词、英文需要先 tokenize），计算其生成概率。 语言模型的评价指标 PPL(困惑度)，用于表示模型生成句子的流利程度。 |

### NLP 核心技术

#### 语义表示

[PaddleLARK](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/pretrain_language_models) 通过在大规模语料上训练得到的通用的语义表示模型，可以助益其他自然语言处理任务，是通用预训练 + 特定任务精调范式的体现。PaddleLARK 集成了 ELMO，BERT，ERNIE 1.0，ERNIE 2.0，XLNet 等热门中英文预训练模型。

| 模型                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [ERNIE](https://github.com/PaddlePaddle/ERNIE)(Enhanced Representation from kNowledge IntEgration) | 百度自研的语义表示模型，通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于 BERT 学习原始语言信号，ERNIE 直接对先验语义知识单元进行建模，增强了模型语义表示能力。 |
| [BERT](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/pretrain_language_models/BERT)(Bidirectional Encoder Representation from Transformers) | 一个迁移能力很强的通用语义表示模型， 以 Transformer 为网络基本组件，以双向 Masked Language Model和 Next Sentence Prediction 为训练目标，通过预训练得到通用语义表示，再结合简单的输出层，应用到下游的 NLP 任务，在多个任务上取得了 SOTA 的结果。 |
| [XLNet](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/pretrain_language_models/XLNet)(XLNet: Generalized Autoregressive Pretraining for Language Understanding) | 重要的语义表示模型之一，引入 Transformer-XL 为骨架，以 Permutation Language Modeling 为优化目标，在若干下游任务上优于 BERT 的性能。 |
| [ELMo](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/pretrain_language_models/ELMo)(Embeddings from Language Models) | 重要的通用语义表示模型之一，以双向 LSTM 为网路基本组件，以 Language Model 为训练目标，通过预训练得到通用的语义表示，将通用的语义表示作为 Feature 迁移到下游 NLP 任务中，会显著提升下游任务的模型性能。 |

#### 文本相似度计算

[SimNet](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/similarity_net) (Similarity Net) 是一个计算短文本相似度的框架，主要包括 BOW、CNN、RNN、MMDNN 等核心网络结构形式。SimNet 框架在百度各产品上广泛应用，提供语义相似度计算训练和预测框架，适用于信息检索、新闻推荐、智能客服等多个应用场景，帮助企业解决语义匹配问题。

#### 文本生成

[seq2seq](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/seq2seq) (Paddle Text Generation) ,一个基于 PaddlePaddle 的文本生成框架，提供了一些列经典文本生成模型案例，如 vanilla seq2seq，seq2seq with attention，variational seq2seq 模型等。

### NLP 系统应用

#### 情感分析

| 模型                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Senta](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/sentiment_classification) (Sentiment Classification，简称Senta) | 面向**通用场景**的情感分类模型，针对带有主观描述的中文文本，可自动判断该文本的情感极性类别。 |
| [EmotionDetection](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/emotion_detection) (Emotion Detection，简称EmoTect) | 专注于识别**人机对话场景**中用户的情绪，针对智能对话场景中的用户文本，自动判断该文本的情绪类别。 |

#### 阅读理解

[machine_reading_comprehension](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/machine_reading_comprehension) (Paddle Machine Reading Comprehension)，集合了百度在阅读理解领域相关的模型，工具，开源数据集等一系列工作。

| 模型                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [DuReader](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/Research/ACL2018-DuReader) | 包含百度开源的基于真实搜索用户行为的中文大规模阅读理解数据集以及基线模型。 |
| [KT-Net](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/Research/ACL2019-KTNET) | 结合知识的阅读理解模型，Squad 曾排名第一。                   |
| [D-Net](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/Research/MRQA2019-D-NET) | 阅读理解十项全能模型，在 EMNLP2019 国际阅读理解大赛夺得 10 项冠军。 |

#### 机器翻译

[machine_translation](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/machine_translation) ，全称为Paddle Machine Translation，基于Transformer的经典机器翻译模型，基于论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

#### 对话系统

[dialogue_system](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/dialogue_system) 包含对话系统方向的模型、数据集和工具。

| 模型                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [DGU](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/dialogue_system/dialogue_general_understanding) (Dialogue General Understanding，通用对话理解模型) | 覆盖了包括**检索式聊天系统**中 context-response matching 任务和**任务完成型对话系统**中**意图识别**，**槽位解析**，**状态追踪**等常见对话系统任务，在 6 项国际公开数据集中都获得了最佳效果。 |
| [ADEM](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/dialogue_system/auto_dialogue_evaluation) (Auto Dialogue Evaluation Model) | 评估开放领域对话系统的回复质量，能够帮助企业或个人快速评估对话系统的回复质量，减少人工评估成本。 |
| [Proactive Conversation](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/Research/ACL2019-DuConv) | 包含百度开源的知识驱动的开放领域对话数据集 [DuConv](https://ai.baidu.com/broad/subordinate?dataset=duconv)，以及基线模型。对应论文 [Proactive Human-Machine Conversation with Explicit Conversation Goals](https://arxiv.org/abs/1906.05572) 发表于 ACL2019。 |
| [DAM](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/Research/ACL2018-DAM)（Deep Attention Matching Network，深度注意力机制模型） | 开放领域多轮对话匹配模型，对应论文 [Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network](https://aclweb.org/anthology/P18-1103/) 发表于 ACL2018。 |

百度最新前沿工作开源，请参考 [Research](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/Research)。

## PaddleRec

个性化推荐，在当前的互联网服务中正在发挥越来越大的作用，目前大部分电子商务系统、社交网络，广告推荐，搜索引擎，都不同程度的使用了各种形式的个性化推荐技术，帮助用户快速找到他们想要的信息。[PaddleRec](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) 包含的模型如下。


| 模型                                                         | 应用场景                       | 简介                                                         |
| :----------------------------------------------------------- | :----------------------------- | :----------------------------------------------------------- |
| [GRU4Rec](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gru4rec) | Session-based 推荐, 图网络推荐 | 首次将 RNN（GRU）运用于 session-based 推荐，核心思想是在一个 session 中，用户点击一系列item的行为看做一个序列，用来训练 RNN 模型 |
| [TagSpace](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/tagspace) | 标签推荐                       | Tagspace 模型学习文本及标签的 embedding 表示，应用于工业级的标签推荐，具体应用场景有 feed 新闻标签推荐。 |
| [SequenceSemanticRetrieval](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ssr) | 召回                           | 解决了 GRU4Rec 模型无法预测训练数据集中不存在的项目，比如新闻推荐的问题。它由两个部分组成：一个是匹配模型部分，另一个是检索部分 |
| [Word2Vec](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/word2vec) | 词向量                         | 训练得到词的向量表示、广泛应用于 NLP 、推荐等任务场景。      |
| [Multiview-Simnet](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/multiview_simnet) | 排序                           | 多视角Simnet模型是可以融合用户以及推荐项目的多个视角的特征并进行个性化匹配学习的一体化模型。这类模型在很多工业化的场景中都会被使用到，比如百度的 Feed 产品中 |
| [GraphNeuralNetwork](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gnn) | 召回                           | SR-GNN，全称为 Session-based Recommendations with Graph Neural Network（GNN）。使用 GNN 进行会话序列建模。 |
| [DeepInterestNetwork](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/din) | 排序                           | DIN，全称为 Deep Interest Network。特点为对历史序列建模的过程中结合了预估目标的信息。 |
| [DeepFM](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/deepfm) | 推荐系统                       | DeepFM，全称 Factorization-Machine based Neural Network。经典的 CTR 推荐算法，网络由DNN和FM两部分组成。 |
| [DCN](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/dcn) | 推荐系统                       | 全称 Deep & Cross Network。提出一种新的交叉网络（cross network），在每个层上明确地应用特征交叉。 |
| [XDeepFM](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/xdeepfm) | 推荐系统                       | xDeepFM，全称 extreme Factorization Machine。对 DeepFM 和 DCN 的改进，提出 CIN（Compressed Interaction Network），使用 vector-wise 等级的显示特征交叉。 |

## PaddleSpeech

[PaddleSpeech](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleSpeech) 包含语音识别和语音合成相关的模型。

| 模型                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [DeepASR](https://github.com/PaddlePaddle/models/blob/release/1.7/PaddleSpeech/DeepASR/README_cn.md) | 利用 PaddlePaddle 框架完成语音识别中声学模型的配置和训练，并集成 Kaldi 的解码器。 |
| [DeepSpeech2](https://github.com/PaddlePaddle/DeepSpeech)    | 一个采用 PaddlePaddle 平台的端到端自动语音识别（ASR）引擎的开源项目，具体原理请参考论文 [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)。 |
| [Parakeet](https://github.com/PaddlePaddle/Parakeet) |定位于灵活、高效的语音合成工具集，支持多个前沿的语音合成模型，包括 WaveFlow、ClariNet、WaveNet、Deep Voice 3、Transformer TTS、FastSpeech 等。 |


## 基于动态图实现的模型

自 PaddlePaddle fluid 1.5 版本正式支持动态图模式以来，模型库新增若干基于动态图实现的模型，请参考 [dygraph](https://github.com/PaddlePaddle/models/blob/develop/dygraph/)，这些模型可以作为了解和使用 PaddlePaddle 动态图模式的示例。目前 PaddlePaddle 的动态图功能正在活跃开发中，API 可能发生变动，欢迎用户试用并给我们反馈。


## 快速下载模型库

由于 github 在国内的下载速度不稳定，我们提供了 models 各版本压缩包的百度云下载地址，以便用户更快速地获取代码。

| 版本号        | tar包                                                         | zip包                                                         |
| ------------- | ------------------------------------------------------------- | ------------------------------------------------------------- |
| models 1.6    | https://paddlepaddle-modles.bj.bcebos.com/models-1.6.tar.gz   | https://paddlepaddle-modles.bj.bcebos.com/models-1.6.zip   |
| models 1.5.1  | https://paddlepaddle-modles.bj.bcebos.com/models-1.5.1.tar.gz | https://paddlepaddle-modles.bj.bcebos.com/models-1.5.1.zip |
| models 1.5    | https://paddlepaddle-modles.bj.bcebos.com/models-1.5.tar.gz   | https://paddlepaddle-modles.bj.bcebos.com/models-1.5.zip   |
| models 1.4    | https://paddlepaddle-modles.bj.bcebos.com/models-1.4.tar.gz   | https://paddlepaddle-modles.bj.bcebos.com/models-1.4.zip   |
| models 1.3    | https://paddlepaddle-modles.bj.bcebos.com/models-1.3.tar.gz   | https://paddlepaddle-modles.bj.bcebos.com/models-1.3.zip   |


## License
This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](LICENSE).


## 许可证书
此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](LICENSE)许可认证。
