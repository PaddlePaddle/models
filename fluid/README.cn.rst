Fluid 模型库
============

图像分类
--------

图像分类是根据图像的语义信息对不同类别图像进行区分，是计算机视觉中重要的基础问题，是物体检测、图像分割、物体跟踪、行为分析、人脸识别等其他高层视觉任务的基础，在许多领域都有着广泛的应用。如：安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

在深度学习时代，图像分类的准确率大幅度提升，在图像分类任务中，我们向大家介绍了如何在经典的数据集ImageNet上，训练常用的模型，包括AlexNet、VGG、GoogLeNet、ResNet、Inception-v4、MobileNet、DPN(Dual
Path
Network)、SE-ResNeXt模型，也开源了\ `训练的模型 <https://github.com/PaddlePaddle/models/blob/develop/fluid/image_classification/README_cn.md#已有模型及其性能>`__\ 方便用户下载使用。同时提供了能够将Caffe模型转换为PaddlePaddle
Fluid模型配置和参数文件的工具。

-  `AlexNet <https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models>`__
-  `VGG <https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models>`__
-  `GoogleNet <https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models>`__
-  `Residual
   Network <https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models>`__
-  `Inception-v4 <https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models>`__
-  `MobileNet <https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models>`__
-  `Dual Path
   Network <https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models>`__
-  `SE-ResNeXt <https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/models>`__
-  `Caffe模型转换为Paddle
   Fluid配置和模型文件工具 <https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification/caffe2fluid>`__

目标检测
--------

目标检测任务的目标是给定一张图像或是一个视频帧，让计算机找出其中所有目标的位置，并给出每个目标的具体类别。对于人类来说，目标检测是一个非常简单的任务。然而，计算机能够“看到”的是图像被编码之后的数字，很难解图像或是视频帧中出现了人或是物体这样的高层语义概念，也就更加难以定位目标出现在图像中哪个区域。与此同时，由于目标会出现在图像或是视频帧中的任何位置，目标的形态千变万化，图像或是视频帧的背景千差万别，诸多因素都使得目标检测对计算机来说是一个具有挑战性的问题。

在目标检测任务中，我们介绍了如何基于\ `PASCAL
VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`__\ 、\ `MS
COCO <http://cocodataset.org/#home>`__\ 数据训练通用物体检测模型，当前介绍了SSD算法，SSD全称Single Shot MultiBox Detector，是目标检测领域较新且效果较好的检测算法之一，具有检测速度快且检测精度高的特点。

开放环境中的检测人脸，尤其是小的、模糊的和部分遮挡的人脸也是一个具有挑战的任务。我们也介绍了如何基于 `WIDER FACE <http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/>`_ 数据训练百度自研的人脸检测PyramidBox模型，该算法于2018年3月份在WIDER FACE的多项评测中均获得 `第一名 <http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html>`_。

-  `Single Shot MultiBox
   Detector <https://github.com/PaddlePaddle/models/blob/develop/fluid/object_detection/README_cn.md>`__
-  `Face Detector: PyramidBox <https://github.com/PaddlePaddle/models/tree/develop/fluid/face_detection/README_cn.md>`_

图像语义分割
------------

图像语意分割顾名思义是将图像像素按照表达的语义含义的不同进行分组/分割，图像语义是指对图像内容的理解，例如，能够描绘出什么物体在哪里做了什么事情等，分割是指对图片中的每个像素点进行标注，标注属于哪一类别。近年来用在无人车驾驶技术中分割街景来避让行人和车辆、医疗影像分析中辅助诊断等。

在图像语义分割任务中，我们介绍如何基于图像级联网络(Image Cascade
Network,ICNet)进行语义分割，相比其他分割算法，ICNet兼顾了准确率和速度。

-  `ICNet <https://github.com/PaddlePaddle/models/tree/develop/fluid/icnet>`__

图像生成
-----------

图像生成是指根据输入向量，生成目标图像。这里的输入向量可以是随机的噪声或用户指定的条件向量。具体的应用场景有：手写体生成、人脸合成、风格迁移、图像修复等。当前的图像生成任务主要是借助生成对抗网络（GAN）来实现。
生成对抗网络（GAN）由两种子网络组成：生成器和识别器。生成器的输入是随机噪声或条件向量，输出是目标图像。识别器是一个分类器，输入是一张图像，输出是该图像是否是真实的图像。在训练过程中，生成器和识别器通过不断的相互博弈提升自己的能力。

在图像生成任务中，我们介绍了如何使用DCGAN和ConditioanlGAN来进行手写数字的生成，另外还介绍了用于风格迁移的CycleGAN.

- `DCGAN & ConditionalGAN <https://github.com/PaddlePaddle/models/tree/develop/fluid/gan/c_gan>`__
- `CycleGAN <https://github.com/PaddlePaddle/models/tree/develop/fluid/gan/cycle_gan>`__

场景文字识别
------------

许多场景图像中包含着丰富的文本信息，对理解图像信息有着重要作用，能够极大地帮助人们认知和理解场景图像的内容。场景文字识别是在图像背景复杂、分辨率低下、字体多样、分布随意等情况下，将图像信息转化为文字序列的过程，可认为是一种特别的翻译过程：将图像输入翻译为自然语言输出。场景图像文字识别技术的发展也促进了一些新型应用的产生，如通过自动识别路牌中的文字帮助街景应用获取更加准确的地址信息等。

在场景文字识别任务中，我们介绍如何将基于CNN的图像特征提取和基于RNN的序列翻译技术结合，免除人工定义特征，避免字符分割，使用自动学习到的图像特征，完成字符识别。当前，介绍了CRNN-CTC模型和基于注意力机制的序列到序列模型。

-  `CRNN-CTC模型 <https://github.com/PaddlePaddle/models/tree/develop/fluid/ocr_recognition>`__
-  `Attention模型 <https://github.com/PaddlePaddle/models/tree/develop/fluid/ocr_recognition>`__


度量学习
-------


度量学习也称作距离度量学习、相似度学习，通过学习对象之间的距离，度量学习能够用于分析对象时间的关联、比较关系，在实际问题中应用较为广泛，可应用于辅助分类、聚类问题，也广泛用于图像检索、人脸识别等领域。以往，针对不同的任务，需要选择合适的特征并手动构建距离函数，而度量学习可根据不同的任务来自主学习出针对特定任务的度量距离函数。度量学习和深度学习的结合，在人脸识别/验证、行人再识别(human Re-ID)、图像检索等领域均取得较好的性能，在这个任务中我们主要介绍了基于Fluid的深度度量学习模型，包含了三元组、四元组等损失函数。

- `Metric Learning <https://github.com/PaddlePaddle/models/tree/develop/fluid/metric_learning>`__


视频分类
-------

视频分类是视频理解任务的基础，与图像分类不同的是，分类的对象不再是静止的图像，而是一个由多帧图像构成的、包含语音数据、包含运动信息等的视频对象，因此理解视频需要获得更多的上下文信息，不仅要理解每帧图像是什么、包含什么，还需要结合不同帧，知道上下文的关联信息。视频分类方法主要包含基于卷积神经网络、基于循环神经网络、或将这两者结合的方法。该任务中我们介绍基于Fluid的视频分类模型，目前包含Temporal Segment Network(TSN)模型，后续会持续增加更多模型。


- `TSN <https://github.com/PaddlePaddle/models/tree/develop/fluid/video_classification>`__



语音识别
--------

自动语音识别（Automatic Speech Recognition,
ASR）是将人类声音中的词汇内容转录成计算机可输入的文字的技术。语音识别的相关研究经历了漫长的探索过程，在HMM/GMM模型之后其发展一直较为缓慢，随着深度学习的兴起，其迎来了春天。在多种语言识别任务中，将深度神经网络(DNN)作为声学模型，取得了比GMM更好的性能，使得
ASR
成为深度学习应用最为成功的领域之一。而由于识别准确率的不断提高，有越来越多的语言技术产品得以落地，例如语言输入法、以智能音箱为代表的智能家居设备等
—— 基于语言的交互方式正在深刻的改变人类的生活。

与 `DeepSpeech <https://github.com/PaddlePaddle/DeepSpeech>`__
中深度学习模型端到端直接预测字词的分布不同，本实例更接近传统的语言识别流程，以音素为建模单元，关注语言识别中声学模型的训练，利用\ `kaldi <http://www.kaldi-asr.org>`__\ 进行音频数据的特征提取和标签对齐，并集成
kaldi 的解码器完成解码。

-  `DeepASR <https://github.com/PaddlePaddle/models/blob/develop/fluid/DeepASR/README_cn.md>`__

机器翻译
--------

机器翻译（Machine
Translation）将一种自然语言(源语言)转换成一种自然语言（目标语音），是自然语言处理中非常基础和重要的研究方向。在全球化的浪潮中，机器翻译在促进跨语言文明的交流中所起的重要作用是不言而喻的。其发展经历了统计机器翻译和基于神经网络的神经机器翻译(Nueural
Machine Translation, NMT)等阶段。在 NMT
成熟后，机器翻译才真正得以大规模应用。而早阶段的 NMT
主要是基于循环神经网络 RNN
的，其训练过程中当前时间步依赖于前一个时间步的计算，时间步之间难以并行化以提高训练速度。因此，非
RNN 结构的 NMT 得以应运而生，例如基于卷积神经网络 CNN
的结构和基于自注意力机制（Self-Attention）的结构。

本实例所实现的 Transformer
就是一个基于自注意力机制的机器翻译模型，其中不再有RNN或CNN结构，而是完全利用
Attention 学习语言中的上下文依赖。相较于RNN/CNN,
这种结构在单层内计算复杂度更低、易于并行化、对长程依赖更易建模，最终在多种语言之间取得了最好的翻译效果。

-  `Transformer <https://github.com/PaddlePaddle/models/blob/develop/fluid/neural_machine_translation/transformer/README_cn.md>`__

强化学习
--------

强化学习是近年来一个愈发重要的机器学习方向，特别是与深度学习相结合而形成的深度强化学习(Deep
Reinforcement Learning,
DRL)，取得了很多令人惊异的成就。人们所熟知的战胜人类顶级围棋职业选手的
AlphaGo 就是 DRL
应用的一个典型例子，除游戏领域外，其它的应用还包括机器人、自然语言处理等。

深度强化学习的开山之作是在Atari视频游戏中的成功应用，
其可直接接受视频帧这种高维输入并根据图像内容端到端地预测下一步的动作，所用到的模型被称为深度Q网络(Deep
Q-Network, DQN)。本实例就是利用PaddlePaddle Fluid这个灵活的框架，实现了
DQN 及其变体，并测试了它们在 Atari 游戏中的表现。

-  `DeepQNetwork <https://github.com/PaddlePaddle/models/blob/develop/fluid/DeepQNetwork/README_cn.md>`__

中文词法分析
------------

中文分词(Word Segmentation)是将连续的自然语言文本，切分出具有语义合理性和完整性的词汇序列的过程。因为在汉语中，词是承担语义的最基本单位，切词是文本分类、情感分析、信息检索等众多自然语言处理任务的基础。 词性标注（Part-of-speech Tagging）是为自然语言文本中的每一个词汇赋予一个词性的过程，这里的词性包括名词、动词、形容词、副词等等。 命名实体识别（Named Entity Recognition，NER）又称作“专名识别”，是指识别自然语言文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等。 我们将这三个任务统一成一个联合任务，称为词法分析任务，基于深度神经网络，利用海量标注语料进行训练，提供了一个端到端的解决方案。

我们把这个联合的中文词法分析解决方案命名为LAC。LAC既可以认为是Lexical Analysis of Chinese的首字母缩写，也可以认为是LAC Analyzes Chinese的递归缩写。

- `LAC <https://github.com/baidu/lac/blob/master/README.md>`__

情感倾向分析
------------

情感倾向分析针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极、 中性。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有力的决策支持。本次我们开放 AI开放平台中情感倾向分析采用的\ `模型 <http://ai.baidu.com/tech/nlp/sentiment_classify>`__\， 提供给用户使用。

- `Senta <https://github.com/baidu/Senta/blob/master/README.md>`__

语义匹配
--------

在自然语言处理很多场景中，需要度量两个文本在语义上的相似度，这类任务通常被称为语义匹配。例如在搜索中根据查询与候选文档的相似度对搜索结果进行排序，文本去重中文本与文本相似度的计算，自动问答中候选答案与问题的匹配等。

本例所开放的DAM (Deep Attention Matching Network)为百度自然语言处理部发表于ACL-2018的工作，用于检索式聊天机器人多轮对话中应答的选择。DAM受Transformer的启发，其网络结构完全基于注意力(attention)机制，利用栈式的self-attention结构分别学习不同粒度下应答和语境的语义表示，然后利用cross-attention获取应答与语境之间的相关性，在两个大规模多轮对话数据集上的表现均好于其它模型。

- `Deep Attention Matching Network <https://github.com/PaddlePaddle/models/tree/develop/fluid/deep_attention_matching_net>`__ 

AnyQ
----

`AnyQ <https://github.com/baidu/AnyQ>`__\ (ANswer Your Questions)
开源项目主要包含面向FAQ集合的问答系统框架、文本语义匹配工具SimNet。
问答系统框架采用了配置化、插件化的设计，各功能均通过插件形式加入，当前共开放了20+种插件。开发者可以使用AnyQ系统快速构建和定制适用于特定业务场景的FAQ问答系统，并加速迭代和升级。

SimNet是百度自然语言处理部于2013年自主研发的语义匹配框架，该框架在百度各产品上广泛应用，主要包括BOW、CNN、RNN、MM-DNN等核心网络结构形式，同时基于该框架也集成了学术界主流的语义匹配模型，如MatchPyramid、MV-LSTM、K-NRM等模型。使用SimNet构建出的模型可以便捷的加入AnyQ系统中，增强AnyQ系统的语义匹配能力。

-  `SimNet in PaddlePaddle
   Fluid <https://github.com/baidu/AnyQ/blob/master/tools/simnet/train/paddle/README.md>`__
   
机器阅读理解
----

机器阅读理解(MRC)是自然语言处理(NLP)中的核心任务之一，最终目标是让机器像人类一样阅读文本，提炼文本信息并回答相关问题。深度学习近年来在NLP中得到广泛使用，也使得机器阅读理解能力在近年有了大幅提高，但是目前研究的机器阅读理解都采用人工构造的数据集，以及回答一些相对简单的问题，和人类处理的数据还有明显差距，因此亟需大规模真实训练数据推动MRC的进一步发展。

百度阅读理解数据集是由百度自然语言处理部开源的一个真实世界数据集，所有的问题、原文都来源于实际数据(百度搜索引擎数据和百度知道问答社区)，答案是由人类回答的。每个问题都对应多个答案，数据集包含200k问题、1000k原文和420k答案，是目前最大的中文MRC数据集。百度同时开源了对应的阅读理解模型，称为DuReader，采用当前通用的网络分层结构，通过双向attention机制捕捉问题和原文之间的交互关系，生成query-aware的原文表示，最终基于query-aware的原文表示通过point network预测答案范围。

-  `DuReader in PaddlePaddle Fluid] <https://github.com/PaddlePaddle/models/blob/develop/fluid/machine_reading_comprehension/README.md>`__
