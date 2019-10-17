# PaddlePaddle Models

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/models) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle provides a rich set of computational units to enable users to adopt a modular approach to solving various learning problems. In this Repo, we demonstrate how to use PaddlePaddle to solve common machine learning tasks, providing several different neural network model that anyone can easily learn and use.

PaddlePaddle 提供了丰富的计算单元，使得用户可以采用模块化的方法解决各种学习问题。在此Repo中，我们展示了如何用 PaddlePaddle来解决常见的机器学习任务，提供若干种不同的易学易用的神经网络模型。PaddlePaddle用户可领取**免费Tesla V100在线算力资源**，高效训练模型，**每日登陆即送12小时**，**连续五天运行再加送48小时**，[前往使用免费算力](http://ai.baidu.com/support/news?action=detail&id=981)。

**目前模型库下模型均要求使用PaddlePaddle 1.6及以上版本或适当的develop版本。**

## 目录
* [智能视觉(PaddleCV)](#PaddleCV)
  * [图像分类](#图像分类)
  * [目标检测](#目标检测)
  * [图像分割](#图像分割)
  * [关键点检测](#关键点检测)
  * [图像生成](#图像生成)
  * [场景文字识别](#场景文字识别)
  * [度量学习](#度量学习)
  * [视频分类和动作定位](#视频分类和动作定位)
* [智能文本处理(PaddleNLP)](#PaddleNLP)
  * [基础模型（词法分析&语言模型）](#基础模型)
  * [文本理解（文本分类&阅读理解）](#文本理解)
  * [语义模型（语义表示&语义匹配）](#语义模型)
  * [文本生成（机器翻译&对话生成）](#文本生成)
* [智能推荐(PaddleRec)](#PaddleRec)
* [其他模型](#其他模型)
* [快速下载模型库](#快速下载模型库)

## PaddleCV

### 图像分类

图像分类是根据图像的语义信息对不同类别图像进行区分，是计算机视觉中重要的基础问题，是物体检测、图像分割、物体跟踪、行为分析、人脸识别等其他高层视觉任务的基础，在许多领域都有着广泛的应用。如：安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

| **模型名称** | **模型简介** | **数据集** | **评估指标 top-1/top-5 accuracy** |
| - | - | - | - |
| [AlexNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | 首次在CNN中成功的应用了ReLU、Dropout和LRN，并使用GPU进行运算加速 | ImageNet-2012验证集 | 56.72%/79.17% |
| [VGG19](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | 在AlexNet的基础上使用3*3小卷积核，增加网络深度，具有很好的泛化能力 | ImageNet-2012验证集 | 72.56%/90.93% |
| [GoogLeNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | 在不增加计算负载的前提下增加了网络的深度和宽度，性能更加优越 | ImageNet-2012验证集 | 70.70%/89.66% |
| [ResNet50](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | Residual Network，引入了新的残差结构，解决了随着网络加深，准确率下降的问题 | ImageNet-2012验证集 | 76.50%/93.00% |
| [ResNet200_vd](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | 融合多种对ResNet改进策略，ResNet200_vd的top1准确率达到80.93% | ImageNet-2012验证集 | 80.93%/95.33% |
| [Inceptionv4](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | 将Inception模块与Residual Connection进行结合，通过ResNet的结构极大地加速训练并获得性能的提升 | ImageNet-2012验证集 | 80.77%/95.26% |
| [MobileNetV1](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | 将传统的卷积结构改造成两层卷积结构的网络，在基本不影响准确率的前提下大大减少计算时间，更适合移动端和嵌入式视觉应用 | ImageNet-2012验证集 | 70.99%/89.68% |
| [MobileNetV2](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | MobileNet结构的微调，直接在thinner的bottleneck层上进行skip learning连接以及对bottleneck layer不进行ReLu非线性处理可取得更好的结果 | ImageNet-2012验证集 | 72.15%/90.65% |
| [SENet154_vd](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | 在ResNeXt 基础、上加入了SE（Sequeeze-and-Excitation）模块，提高了识别准确率，在ILSVRC 2017 的分类项目中取得了第一名 | ImageNet-2012验证集 | 81.40%/95.48% |
| [ShuffleNetV2](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) | ECCV2018，轻量级CNN网络，在速度和准确度之间做了很好地平衡。在同等复杂度下，比ShuffleNet和MobileNetv2更准确，更适合移动端以及无人车领域 | ImageNet-2012验证集 | 70.03%/89.17% |

更多图像分类模型请参考[Image Classification](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification)

### 目标检测

目标检测任务的目标是给定一张图像或是一个视频帧，让计算机找出其中所有目标的位置，并给出每个目标的具体类别。对于计算机而言，能够“看到”的是图像被编码之后的数字，但很难解图像或是视频帧中出现了人或是物体这样的高层语义概念，也就更加难以定位目标出现在图像中哪个区域。

| 模型名称                                                     | 模型简介                                                     | 数据集     | 评估指标   mAP                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- | ------------------------------------------------------- |
| [SSD](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection) | 很好的继承了MobileNet预测速度快，易于部署的特点，能够很好的在多种设备上完成图像目标检测任务 | VOC07 test | mAP   = 73.32%                                          |
| [Faster-RCNN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection) | 创造性地采用卷积网络自行产生建议框，并且和目标检测网络共享卷积网络，建议框数目减少，质量提高 | MS-COCO    | 基于ResNet 50  mAP(0.50:0.95) = 36.7%                   |
| [Mask-RCNN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection) | 经典的两阶段框架，在Faster R-CNN模型基础上添加分割分支，得到掩码结果，实现了掩码和类别预测关系的解藕，可得到像素级别的检测结果。 | MS-COCO    | 基于ResNet 50   Mask   mAP（0.50:0.95） = 31.4%         |
| [RetinaNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection) | 经典的一阶段框架，由主干网络、FPN结构、和两个分别用于回归物体位置和预测物体类别的子网络组成。在训练过程中使用Focal Loss，解决了传统一阶段检测器存在前景背景类别不平衡的问题，进一步提高了一阶段检测器的精度。 | MS-COCO    | 基于ResNet 50 mAP (0.50:0.95) = 36%                      |
| [YOLOv3](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleDetection) | 速度和精度均衡的目标检测网络，相比于原作者darknet中的YOLO v3实现，PaddlePaddle实现参考了论文[Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf) 增加了mixup，label_smooth等处理，精度(mAP(0.50：0.95))相比于原作者提高了4.7个绝对百分点，在此基础上加入synchronize batch   normalization, 最终精度相比原作者提高5.9个绝对百分点。 | MS-COCO    | 基于DarkNet   mAP(0.50:0.95)=   38.9%                   |
| [PyramidBox](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/face_detection) | **PyramidBox** **模型是百度自主研发的人脸检测模型**，利用上下文信息解决困难人脸的检测问题，网络表达能力高，鲁棒性强。于18年3月份在WIDER Face数据集上取得第一名 | WIDER FACE | mAP   （Easy/Medium/Hard   set）=   96.0%/ 94.8%/ 88.8% |

### 图像分割

图像语义分割顾名思义是将图像像素按照表达的语义含义的不同进行分组/分割，图像语义是指对图像内容的理解，例如，能够描绘出什么物体在哪里做了什么事情等，分割是指对图片中的每个像素点进行标注，标注属于哪一类别。近年来用在无人车驾驶技术中分割街景来避让行人和车辆、医疗影像分析中辅助诊断等。

| 模型名称                                                     | 模型简介                                                     | 数据集    | 评估指标        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------- | --------------- |
| [ICNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/icnet) | 主要用于图像实时语义分割，能够兼顾速度和准确性，易于线上部署 | Cityscape | Mean IoU=67.0%  |
| [DeepLab   V3+](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/deeplabv3%2B) | 通过encoder-decoder进行多尺度信息的融合，同时保留了原来的空洞卷积和ASSP层，   其骨干网络使用了Xception模型，提高了语义分割的健壮性和运行速率 | Cityscape | Mean IoU=78.81% |

### 关键点检测

人体骨骼关键点检测，Pose Estimation，主要检测人体的一些关键点，如关节，五官等，通过关键点描述人体骨骼信息。人体骨骼关键点检测对于描述人体姿态，预测人体行为至关重要。是诸多计算机视觉任务的基础，例如动作分类，异常行为检测，以及自动驾驶等等。

| 模型名称                                                     | 模型简介                                                     | 数据集       | 评估指标     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ | ------------ |
| [Simple   Baselines](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/human_pose_estimation) | coco2018关键点检测项目亚军方案，网络结构非常简单，效果达到state of the art | COCO val2017 | AP =   72.7% |

### 图像生成

图像生成是指根据输入向量，生成目标图像。这里的输入向量可以是随机的噪声或用户指定的条件向量。具体的应用场景有：手写体生成、人脸合成、风格迁移、图像修复等。

| 模型名称                                                     | 模型简介                                                     | 数据集     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- |
| [CGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | 条件生成对抗网络，一种带条件约束的GAN，使用额外信息对模型增加条件，可以指导数据生成过程 | Mnist      |
| [DCGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | 深度卷积生成对抗网络，将GAN和卷积网络结合起来，以解决GAN训练不稳定的问题 | Mnist      |
| [Pix2Pix](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | 图像翻译，通过成对图片将某一类图片转换成另外一类图片，可用于风格迁移 | Cityscapes |
| [CycleGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | 图像翻译，可以通过非成对的图片将某一类图片转换成另外一类图片，可用于风格迁移 | Cityscapes |
| [StarGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | 多领域属性迁移，引入辅助分类帮助单个判别器判断多个属性，可用于人脸属性转换 | Celeba     |
| [AttGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | 利用分类损失和重构损失来保证改变特定的属性，可用于人脸特定属性转换 | Celeba     |
| [STGAN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleGAN) | 人脸特定属性转换，只输入有变化的标签，引入GRU结构，更好的选择变化的属性 | Celeba     |

### 场景文字识别

场景文字识别是在图像背景复杂、分辨率低下、字体多样、分布随意等情况下，将图像信息转化为文字序列的过程，可认为是一种特别的翻译过程：将图像输入翻译为自然语言输出。

| 模型名称                                                     | 模型简介                                                     | 数据集                     | 评估指标       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------- | -------------- |
| [CRNN-CTC](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/ocr_recognition) | 使用CTC model识别图片中单行英文字符，用于端到端的文本行图片识别方法 | 单行不定长的英文字符串图片 | 错误率= 22.3%  |
| [OCR   Attention](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/ocr_recognition) | 使用attention 识别图片中单行英文字符，用于端到端的自然场景文本识别， | 单行不定长的英文字符串图片 | 错误率 = 15.8% |

### 度量学习

度量学习也称作距离度量学习、相似度学习，通过学习对象之间的距离，度量学习能够用于分析对象时间的关联、比较关系，在实际问题中应用较为广泛，可应用于辅助分类、聚类问题，也广泛用于图像检索、人脸识别等领域。

| 模型名称                                                     | 模型简介                                                  | 数据集                         | 评估指标   Recall@Rank-1（使用arcmargin训练） |
| ------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------ | --------------------------------------------- |
| [ResNet50未微调](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning) | 使用arcmargin loss训练的特征模型                          | Stanford   Online Product(SOP) | 78.11%                                        |
| [ResNet50使用triplet微调](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning) | 在arcmargin loss基础上，使用triplet loss微调的特征模型    | Stanford   Online Product(SOP) | 79.21%                                        |
| [ResNet50使用quadruplet微调](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning) | 在arcmargin loss基础上，使用quadruplet loss微调的特征模型 | Stanford   Online Product(SOP) | 79.59%                                        |
| [ResNet50使用eml微调](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning) | 在arcmargin loss基础上，使用eml loss微调的特征模型        | Stanford   Online Product(SOP) | 80.11%                                        |
| [ResNet50使用npairs微调](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning) | 在arcmargin loss基础上，使用npairs loss微调的特征模型     | Stanford   Online Product(SOP) | 79.81%                                        |

### 视频分类和动作定位

视频分类和动作定位是视频理解任务的基础。视频数据包含语音、图像等多种信息，因此理解视频任务不仅需要处理语音和图像，还需要提取视频帧时间序列中的上下文信息。视频分类模型提供了提取全局时序特征的方法，主要方式有卷积神经网络(C3D,I3D,C2D等)，神经网络和传统图像算法结合(VLAD等)，循环神经网络等建模方法。视频动作定位模型需要同时识别视频动作的类别和起止时间点，通常采用类似于图像目标检测中的算法在时间维度上进行建模。

| 模型名称                                                     | 模型简介                                                     | 数据集                     | 评估指标    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------- | ----------- |
| [TSN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | ECCV'16提出的基于2D-CNN经典解决方案 | Kinetics-400               | Top-1 = 67% |
| [Non-Local](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 视频非局部关联建模模型 | Kinetics-400               | Top-1 = 74% |
| [stNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | AAAI'19提出的视频联合时空建模方法 | Kinetics-400               | Top-1 = 69% |
| [TSM](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 基于时序移位的简单高效视频时空建模方法 | Kinetics-400               | Top-1 = 70% |
| [Attention   LSTM](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 常用模型，速度快精度高 | Youtube-8M                 | GAP   = 86% |
| [Attention   Cluster](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | CVPR'18提出的视频多模态特征注意力聚簇融合方法 | Youtube-8M                 | GAP   = 84% |
| [NeXtVlad](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 2nd-Youtube-8M比赛第3名的模型 | Youtube-8M                 | GAP   = 87% |
| [C-TCN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 2018年ActivityNet夺冠方案 | ActivityNet1.3 | MAP=31%    |
| [BSN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 为视频动作定位问题提供高效的proposal生成方法 | ActivityNet1.3 | AUC=66.64%    |
| [BMN](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo) | 2019年ActivityNet夺冠方案 | ActivityNet1.3 | AUC=67.19%    |

## PaddleNLP

### 基础模型

#### 词法分析

[LAC(Lexical Analysis of Chinese)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/lexical_analysis)百度自主研发中文特色模型词法分析任务，**输入是一个字符串，而输出是句子中的词边界和词性、实体类别。

| **模型**         | **Precision** | **Recall** | **F1-score** |
| ---------------- | ------------- | ---------- | ------------ |
| Lexical Analysis | 88.0%         | 88.7%      | 88.4%        |
| BERT finetuned   | 90.2%         | 90.4%      | 90.3%        |
| ERNIE finetuned  | 92.0%         | 92.0%      | 92.0%        |

#### 语言模型

[基于LSTM的语言模型任务](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/language_model)，给定一个输入词序列（中文分词、英文tokenize），计算其PPL（语言模型困惑度，用户表示句子的流利程度）。

| **large config** | **train** | **valid** | **test** |
| ---------------- | --------- | --------- | -------- |
| paddle           | 37.221    | 82.358    | 78.137   |
| tensorflow       | 38.342    | 82.311    | 78.121   |

### 文本理解

#### 情感分析

[Senta(Sentiment Classification)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/sentiment_classification)百度AI开放平台中情感倾向分析模型、百度自主研发的中文情感分析特色模型。

| **模型**      | **dev** | **test** | **模型（****finetune****）** | **dev** | **test** |
| ------------- | ------- | -------- | ---------------------------- | ------- | -------- |
| BOW           | 89.8%   | 90.0%    | BOW                          | 91.3%   | 90.6%    |
| CNN           | 90.6%   | 89.9%    | CNN                          | 92.4%   | 91.8%    |
| LSTM          | 90.0%   | 91.0%    | LSTM                         | 93.3%   | 92.2%    |
| GRU           | 90.0%   | 89.8%    | GRU                          | 93.3%   | 93.2%    |
| BI-LSTM       | 88.5%   | 88.3%    | BI-LSTM                      | 92.8%   | 91.4%    |
| ERNIE         | 95.1%   | 95.4%    | ERNIE                        | 95.4%   | 95.5%    |
| ERNIE+BI-LSTM | 95.3%   | 95.2%    | ERNIE+BI-LSTM                | 95.7%   | 95.6%    |

#### 对话情绪识别

[EmoTect(Emotion Detection)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/emotion_detection)专注于识别智能对话场景中用户的情绪识别，并开源基于百度海量数据训练好的预训练模型。

| **模型** | **闲聊** | **客服** | **微博** |
| -------- | -------- | -------- | -------- |
| BOW      | 90.2%    | 87.6%    | 74.2%    |
| LSTM     | 91.4%    | 90.1%    | 73.8%    |
| Bi-LSTM  | 91.2%    | 89.9%    | 73.6%    |
| CNN      | 90.8%    | 90.7%    | 76.3%    |
| TextCNN  | 91.1%    | 91.0%    | 76.8%    |
| BERT     | 93.6%    | 92.3%    | 78.6%    |
| ERNIE    | 94.4%    | 94.0%    | 80.6%    |

#### 阅读理解

[MRC(Machine Reading Comprehension)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2018-DuReader)机器阅读理解(MRC)是自然语言处理(NLP)中的关键任务之一，开源的DuReader升级了经典的阅读理解BiDAF模型，去掉了char级别的embedding，在预测层中使用了[pointer network](https://arxiv.org/abs/1506.03134)，并且参考了[R-NET](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)中的一些网络结构，效果上有了大幅提升

| **Model**                                                | **Dev ROUGE-L** | **Test ROUGE-L** |
| -------------------------------------------------------- | --------------- | ---------------- |
| BiDAF (原始[论文](https://arxiv.org/abs/1711.05073)基线) | 39.29           | 45.90            |
| 本基线系统                                               | 47.68           | 54.66            |

### 语义模型

#### ERNIE

[ERNIE(Enhanced Representation from kNowledge IntEgration)](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)百度自研的语义表示模型，通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于 BERT 学习原始语言信号，ERNIE直接对先验语义知识单元进行建模，增强了模型语义表示能力。

<table border="1" cellspacing="0" cellpadding="0" width="0">
  <tr>
    <td width="66"><p align="center">数据集 </p></td>
    <td width="180" colspan="2"><p align="center">XNLI</p></td>
    <td width="196" colspan="2"><p align="center">LCQMC</p></td>
    <td width="196" colspan="2"><p align="center">MSRA-NER<br />
        (SIGHAN 2006)</p></td>
    <td width="196" colspan="2"><p align="center">ChnSentiCorp</p></td>
    <td width="392" colspan="4"><p align="center">nlpcc-dbqa</p></td>
  </tr>
  <tr>
    <td width="66" rowspan="2"><p align="center">评估<br />指标</p></td>
    <td width="180" colspan="2"><p align="center">acc</p></td>
    <td width="196" colspan="2"><p align="center">acc</p></td>
    <td width="196" colspan="2"><p align="center">f1-score</p></td>
    <td width="196" colspan="2"><p align="center">acc</p></td>
    <td width="196" colspan="2"><p align="center">mrr</p></td>
    <td width="196" colspan="2"><p align="center">f1-score</p></td>
  </tr>
  <tr>
    <td width="82"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
    <td width="98"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
    <td width="98"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
    <td width="98"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
    <td width="98"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
    <td width="98"><p align="center">dev</p></td>
    <td width="98"><p align="center">test</p></td>
  </tr>
  <tr>
    <td width="66"><p align="center">BERT</p></td>
    <td width="82"><p align="center">78.1</p></td>
    <td width="98"><p align="center">77.2</p></td>
    <td width="98"><p align="center">88.8</p></td>
    <td width="98"><p align="center">87</p></td>
    <td width="98"><p align="center">94.0</p></td>
    <td width="98"><p align="center">92.6</p></td>
    <td width="98"><p align="center">94.6</p></td>
    <td width="98"><p align="center">94.3</p></td>
    <td width="98"><p align="center">94.7</p></td>
    <td width="98"><p align="center">94.6</p></td>
    <td width="98"><p align="center">80.7</p></td>
    <td width="98"><p align="center">80.8</p></td>
  </tr>
  <tr>
    <td width="66"><p align="center">ERNIE</p></td>
    <td width="82"><p>79.9(+1.8)</p></td>
    <td width="98"><p>78.4(+1.2)</p></td>
    <td width="98"><p>89.7(+0.9)</p></td>
    <td width="98"><p>87.4(+0.4)</p></td>
    <td width="98"><p>95.0(+1.0)</p></td>
    <td width="98"><p>93.8(+1.2)</p></td>
    <td width="98"><p>95.2(+0.6)</p></td>
    <td width="98"><p>95.4(+1.1)</p></td>
    <td width="98"><p>95.0(+0.3)</p></td>
    <td width="98"><p>95.1(+0.5)</p></td>
    <td width="98"><p>82.3(+1.6)</p></td>
    <td width="98"><p>82.7(+1.9)</p></td>
  </tr>
</table>

#### BERT

[BERT(Bidirectional Encoder Representation from Transformers)](https://github.com/PaddlePaddle/LARK/tree/develop/BERT)是一个迁移能力很强的通用语义表示模型， 以 Transformer 为网络基本组件，以双向 Masked Language Model和 Next Sentence Prediction 为训练目标，通过预训练得到通用语义表示，再结合简单的输出层，应用到下游的 NLP 任务，在多个任务上取得了 SOTA 的结果。

#### ELMo

[ELMo(Embeddings from Language Models)](https://github.com/PaddlePaddle/LARK/tree/develop/ELMo)是重要的通用语义表示模型之一，以双向 LSTM 为网路基本组件，以 Language Model 为训练目标，通过预训练得到通用的语义表示，将通用的语义表示作为 Feature 迁移到下游 NLP 任务中，会显著提升下游任务的模型性能。

#### SimNet

[SimNet(Similarity Net)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/similarity_net)一个计算短文本相似度的框架，可以根据用户输入的两个文本，计算出相似度得分。

| **模型**     | **百度知道** | **ECOM** | **QQSIM** | **UNICOM** | **LCQMC** |
| ------------ | ------------ | -------- | --------- | ---------- | --------- |
|              | AUC          | AUC      | AUC       | 正逆序比   | Accuracy  |
| BOW_Pairwise | 0.6767       | 0.7329   | 0.7650    | 1.5630     | 0.7532    |

### 文本生成

#### 机器翻译

[MT(machine translation)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/neural_machine_translation/transformer)机器翻译是利用计算机将一种自然语言(源语言)转换为另一种自然语言(目标语言)的过程，输入为源语言句子，输出为相应的目标语言的句子。

| **测试集** | **newstest2014** | **newstest2015** | **newstest2016** |
| ---------- | ---------------- | ---------------- | ---------------- |
| Base       | 26.35            | 29.07            | 33.30            |
| Big        | 27.07            | 30.09            | 34.38            |

#### 对话自动评估

[对话自动评估(Auto Dialogue Evaluation)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/dialogue_model_toolkit/auto_dialogue_evaluation)主要用于评估开放领域对话系统的回复质量，能够帮助企业或个人快速评估对话系统的回复质量，减少人工评估成本。

利用少量标注数据微调后，自动评估打分和人工打分spearman相关系数，如下表。

| **/** | **seq2seq_naive** | **seq2seq_att** | **keywords** | **human** |
| ----- | ----------------- | --------------- | ------------ | --------- |
| cor   | 0.474             | 0.477           | 0.443        | 0.378     |

#### 对话通用理解

[DGU(Dialogue General Understanding)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/dialogue_model_toolkit/dialogue_general_understanding)对话通用理解针对数据集开发了相关的模型训练过程，支持分类，多标签分类，序列标注等任务，用户可针对自己的数据集，进行相关的模型定制

| **ask_name** | **udc** | **udc** | **udc** | **atis_slot** | **dstc2**  | **atis_intent** | **swda** | **mrda** |
| ------------ | ------- | ------- | ------- | ------------- | ---------- | --------------- | -------- | -------- |
| 对话任务     | 匹配    | 匹配    | 匹配    | 槽位解析      | DST        | 意图识别        | DA       | DA       |
| 任务类型     | 分类    | 分类    | 分类    | 序列标注      | 多标签分类 | 分类            | 分类     | 分类     |
| 任务名称     | udc     | udc     | udc     | atis_slot     | dstc2      | atis_intent     | swda     | mrda     |
| 评估指标     | R1@10   | R2@10   | R5@10   | F1            | JOINT ACC  | ACC             | ACC      | ACC      |
| SOTA         | 76.70%  | 87.40%  | 96.90%  | 96.89%        | 74.50%     | 98.32%          | 81.30%   | 91.70%   |
| DGU          | 82.02%  | 90.43%  | 97.75%  | 97.10%        | 89.57%     | 97.65%          | 80.19%   | 91.43%   |

#### DAM

[深度注意力机制模型(Deep Attention Maching)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/Research/ACL2018-DAM)是开放领域多轮对话匹配模型。根据多轮对话历史和候选回复内容，排序出最合适的回复。

|      | Ubuntu Corpus | Douban Conversation Corpus |       |       |       |       |       |       |       |       |
| ---- | ------------- | -------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|      | R2@1          | R10@1                      | R10@2 | R10@5 | MAP   | MRR   | P@1   | R10@1 | R10@2 | R10@5 |
| DAM  | 93.8%         | 76.7%                      | 87.4% | 96.9% | 55.0% | 60.1% | 42.7% | 25.4% | 41.0% | 75.7% |

#### 知识驱动对话

[知识驱动对话的新对话任务](https://github.com/baidu/knowledge-driven-dialogue/tree/master)其中机器基于构建的知识图与人交谈。它旨在测试机器进行类似人类对话的能力。

| **baseline system** | **F1/BLEU1/BLEU2** | **DISTINCT1/DISTINCT2** |
| ------------------- | ------------------ | ----------------------- |
| retrieval-based     | 31.72/0.291/0.156  | 0.118/0.373             |
| generation-based    | 32.65/0.300/0.168  | 0.062/0.128             |

## PaddleRec

个性化推荐，在当前的互联网服务中正在发挥越来越大的作用，目前大部分电子商务系统、社交网络，广告推荐，搜索引擎，都不同程度的使用了各种形式的个性化推荐技术，帮助用户快速找到他们想要的信息。

| 模型名称                                                     | 模型简介                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [TagSpace](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | 应用于工业级的标签推荐，具体应用场景有feed新闻标签推荐等     |
| [GRU4Rec](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | 首次将RNN（GRU）运用于session-based推荐，相比传统的KNN和矩阵分解，效果有明显的提升 |
| [SequenceSemanticRetrieval](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | 使用参考论文中的思想，使用多种时间粒度进行用户行为预测       |
| [DeepCTR](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | 只实现了DeepFM论文中介绍的模型的DNN部分，DeepFM会在其他例子中给出 |
| [Multiview-Simnet](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | 基于多元视图，将用户和项目的多个功能视图合并为一个统一模型   |
| [Word2Vec](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | skip-gram模式的word2vector模型                               |
| [GraphNeuralNetwork](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | 基于会话的图神经网络模型的推荐系统，可以更好的挖掘item中丰富的转换特性以及生成准确的潜在的用户向量表示 |
| [DeepInterestNetwork](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec) | DIN通过一个兴趣激活模块(Activation Unit)，用预估目标Candidate ADs的信息去激活用户的历史点击商品，以此提取用户与当前预估目标相关的兴趣。 |


## 其他模型

| 模型名称                                                     | 模型简介                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [DeepASR](https://github.com/PaddlePaddle/models/blob/develop/PaddleSpeech/DeepASR/README_cn.md) | 利用Fluid框架完成语音识别中声学模型的配置和训练，并集成 Kaldi 的解码器 |
| [DQN](https://github.com/PaddlePaddle/models/blob/develop/legacy/PaddleRL/DeepQNetwork/README_cn.md) | value   based强化学习算法，第一个成功地将深度学习和强化学习结合起来的模型 |
| [DoubleDQN](https://github.com/PaddlePaddle/models/blob/develop/legacy/PaddleRL/DeepQNetwork/README_cn.md) | 将Double Q的想法应用在DQN上，解决过优化问题                  |
| [DuelingDQN](https://github.com/PaddlePaddle/models/blob/develop/legacy/PaddleRL/DeepQNetwork/README_cn.md) | 改进了DQN模型，提高了模型的性能                              |

## 快速下载模型库

由于github在国内的下载速度不稳定，我们提供了models各版本压缩包的百度云下载地址，以便用户更快速的获取代码。

| 版本号        | tar包                                                         | zip包                                                         |
| ------------- | ------------------------------------------------------------- | ------------------------------------------------------------- |
| models 1.5.1  | https://paddlepaddle-modles.bj.bcebos.com/models-1.5.1.tar.gz | https://paddlepaddle-modles.bj.bcebos.com/models-1.5.1.zip |
| models 1.5    | https://paddlepaddle-modles.bj.bcebos.com/models-1.5.tar.gz   | https://paddlepaddle-modles.bj.bcebos.com/models-1.5.zip   |
| models 1.4    | https://paddlepaddle-modles.bj.bcebos.com/models-1.4.tar.gz   | https://paddlepaddle-modles.bj.bcebos.com/models-1.4.zip   |
| models 1.3    | https://paddlepaddle-modles.bj.bcebos.com/models-1.3.tar.gz   | https://paddlepaddle-modles.bj.bcebos.com/models-1.3.zip   |


## License
This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](LICENSE).


## 许可证书
此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](LICENSE)许可认证。
