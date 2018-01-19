# models 简介

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/models)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://github.com/PaddlePaddle/models)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle提供了丰富的运算单元，帮助大家以模块化的方式构建起千变万化的深度学习模型来解决不同的应用问题。这里，我们针对常见的机器学习任务，提供了不同的神经网络模型供大家学习和使用。


## 1. 词向量

词向量用一个实向量表示词语，向量的每个维都表示文本的某种潜在语法或语义特征，是深度学习应用于自然语言处理领域最成功的概念和成果之一。广义的，词向量也可以应用于普通离散特征。词向量的学习通常都是一个无监督的学习过程，因此，可以充分利用海量的无标记数据以捕获特征之间的关系，也可以有效地解决特征稀疏、标签数据缺失、数据噪声等问题。然而，在常见词向量学习方法中，模型最后一层往往会遇到一个超大规模的分类问题，是计算性能的瓶颈。

在词向量任务中，我们向大家展示如何使用Hierarchical-Sigmoid 和噪声对比估计（Noise Contrastive Estimation，NCE）来加速词向量的学习。

- 1.1 [Hsigmoid加速词向量训练](https://github.com/PaddlePaddle/models/tree/develop/hsigmoid)
- 1.2 [噪声对比估计加速词向量训练](https://github.com/PaddlePaddle/models/tree/develop/nce_cost)


## 2. RNN 语言模型

语言模型是自然语言处理领域里一个重要的基础模型，除了得到词向量（语言模型训练的副产物），还可以帮助我们生成文本。给定若干个词，语言模型可以帮助我们预测下一个最可能出现的词。

在利用语言模型生成文本的任务中，我们重点介绍循环神经网络语言模型，大家可以通过文档中的使用说明快速适配到自己的训练语料，完成自动写诗、自动写散文等有趣的模型。

- 2.1 [使用循环神经网络语言模型生成文本](https://github.com/PaddlePaddle/models/tree/develop/generate_sequence_by_rnn_lm)

## 3. 点击率预估

点击率预估模型预判用户对一条广告点击的概率，对每次广告的点击情况做出预测，是广告技术的核心算法之一。逻谛斯克回归对大规模稀疏特征有着很好的学习能力，在点击率预估任务发展的早期一统天下。近年来，DNN 模型由于其强大的学习能力逐渐接过点击率预估任务的大旗。

在点击率预估任务中，我们首先给出谷歌提出的 Wide & Deep 模型。这一模型融合了适用于学习抽象特征的DNN和适用于大规模稀疏特征的逻谛斯克回归两者的优点，可以作为一种相对成熟的模型框架使用，在工业界也有一定的应用。同时，我们提供基于因子分解机的深度神经网络模型，该模型融合了因子分解机和深度神经网络，分别建模输入属性之间的低阶交互和高阶交互。

- 3.1 [Wide & deep 点击率预估模型](https://github.com/PaddlePaddle/models/tree/develop/ctr/README.cn.md)
- 3.2 [基于深度因子分解机的点击率预估模型](https://github.com/PaddlePaddle/models/tree/develop/deep_fm)

## 4. 文本分类

文本分类是自然语言处理领域最基础的任务之一，深度学习方法能够免除复杂的特征工程，直接使用原始文本作为输入，数据驱动地最优化分类准确率。

在文本分类任务中，我们以情感分类任务为例，提供了基于DNN的非序列文本分类模型，以及基于CNN的序列模型供大家学习和使用（基于LSTM的模型见PaddleBook中[情感分类](http://www.paddlepaddle.org/docs/develop/book/06.understand_sentiment/index.cn.html)一课）。

- 4.1 [基于DNN/CNN的情感分类](https://github.com/PaddlePaddle/models/tree/develop/text_classification)
- 4.2 [基于双层序列的文本分类模型](https://github.com/PaddlePaddle/models/tree/develop/nested_sequence/text_classification)

## 5. 排序学习

排序学习(Learning to Rank， LTR)是信息检索和搜索引擎研究的核心问题之一，通过机器学习方法学习一个分值函数对待排序的候选进行打分，再根据分值的高低确定序关系。深度神经网络可以用来建模分值函数，构成各类基于深度学习的LTR模型。

在排序学习任务中，我们介绍基于RankLoss损失函数Pairwise排序模型和基于LambdaRank损失函数的Listwise排序模型(Pointwise学习策略见PaddleBook中[推荐系统](http://www.paddlepaddle.org/docs/develop/book/05.recommender_system/index.cn.html)一课）。

- 5.1 [基于Pairwise和Listwise的排序学习](https://github.com/PaddlePaddle/models/tree/develop/ltr)

## 6. 结构化语义模型

深度结构化语义模型是一种基于神经网络的语义匹配模型框架，可以用于学习两路信息实体或是文本之间的语义相似性。DSSM使用DNN、CNN或是RNN将两路信息实体或是文本映射到同一个连续的低纬度语义空间中。在这个语义空间中，两路实体或是文本可以同时进行表示，然后，通过定义距离度量和匹配函数来刻画并学习不同实体或是文本在同一个语义空间内的语义相似性。

在结构化语义模型任务中，我们演示如何建模两个字符串之间的语义相似度。模型支持DNN(全连接前馈网络)、CNN(卷积网络)、RNN(递归神经网络)等不同的网络结构，以及分类、回归、排序等不同损失函数。本例采用最简单的文本数据作为输入，通过替换自己的训练和预测数据，便可以在真实场景中使用。

- 6.1 [深度结构化语义模型](https://github.com/PaddlePaddle/models/tree/develop/dssm/README.cn.md)

## 7. 命名实体识别

给定输入序列，序列标注模型为序列中每一个元素贴上一个类别标签，是自然语言处理领域最基础的任务之一。随着深度学习方法的不断发展，利用循环神经网络学习输入序列的特征表示，条件随机场（Conditional Random Field, CRF）在特征基础上完成序列标注任务，逐渐成为解决序列标注问题的标配解决方案。

在序列标注任务中，我们以命名实体识别（Named Entity Recognition，NER）任务为例，介绍如何训练一个端到端的序列标注模型。

- 7.1 [命名实体识别](https://github.com/PaddlePaddle/models/tree/develop/sequence_tagging_for_ner)

## 8. 序列到序列学习

序列到序列学习实现两个甚至是多个不定长模型之间的映射，有着广泛的应用，包括：机器翻译、智能对话与问答、广告创意语料生成、自动编码（如金融画像编码）、判断多个文本串之间的语义相关性等。

在序列到序列学习任务中，我们首先以机器翻译任务为例，提供了多种改进模型供大家学习和使用。包括：不带注意力机制的序列到序列映射模型，这一模型是所有序列到序列学习模型的基础；使用Scheduled Sampling改善RNN模型在生成任务中的错误累积问题；带外部记忆机制的神经机器翻译，通过增强神经网络的记忆能力，来完成复杂的序列到序列学习任务。除机器翻译任务之外，我们也提供了一个基于深层LSTM网络生成古诗词，实现同语言生成的模型。

- 8.1 [无注意力机制的神经机器翻译](https://github.com/PaddlePaddle/models/tree/develop/nmt_without_attention/README.cn.md)
- 8.2 [使用Scheduled Sampling改善翻译质量](https://github.com/PaddlePaddle/models/tree/develop/scheduled_sampling)
- 8.3 [带外部记忆机制的神经机器翻译](https://github.com/PaddlePaddle/models/tree/develop/mt_with_external_memory)
- 8.4 [生成古诗词](https://github.com/PaddlePaddle/models/tree/develop/generate_chinese_poetry)

## 9. 阅读理解

当深度学习以及各类新技术不断推动自然语言处理领域向前发展时，我们不禁会问：应该如何确认模型真正理解了人类特有的自然语言，具备一定的理解和推理能力？纵观NLP领域的各类经典问题：词法分析、句法分析、情感分类、写诗等，这些问题的经典解决方案，从技术原理上距离“语言理解”仍有一定距离。为了衡量现有NLP技术到“语言理解”这一终极目标之间的差距，我们需要一个有足够难度且可量化可复现的任务，这也是阅读理解问题提出的初衷。尽管目前的研究现状表明在现有阅读理解数据集上表现良好的模型，依然没有做到真正的语言理解，但机器阅读理解依然被视为是检验模型向理解语言迈进的一个重要任务。

阅读理解本质上也是自动问答的一种，模型“阅读”一段文字后回答给定的问题，在这一任务中，我们介绍使用Learning to Search 方法，将阅读理解转化为从段落中寻找答案所在句子，答案在句子中的起始位置，以及答案在句子中的结束位置，这样一个多步决策过程。

- 9.1 [Globally Normalized Reader](https://github.com/PaddlePaddle/models/tree/develop/globally_normalized_reader)

## 10. 自动问答

自动问答（Question Answering）系统利用计算机自动回答用户提出的问题，是验证机器是否具备自然语言理解能力的重要任务之一，其研究历史可以追溯到人工智能的原点。与检索系统相比，自动问答系统是信息服务的一种高级形式，系统返回给用户的不再是排序后的基于关键字匹配的检索结果，而是精准的自然语言答案。

在自动问答任务中，我们介绍基于深度学习的端到端问答系统，将自动问答转化为一个序列标注问题。端对端问答系统试图通过从高质量的"问题-证据（Evidence）-答案"数据中学习，建立一个联合学习模型，同时学习语料库、知识库、问句语义表示之间的语义映射关系，将传统的问句语义解析、文本检索、答案抽取与生成的复杂步骤转变为一个可学习过程。

- 10.1 [基于序列标注的事实型自动问答模型](https://github.com/PaddlePaddle/models/tree/develop/neural_qa)

## 11. 图像分类

图像相比文字能够提供更加生动、容易理解及更具艺术感的信息，是人们转递与交换信息的重要来源。图像分类是根据图像的语义信息对不同类别图像进行区分，是计算机视觉中重要的基础问题，也是图像检测、图像分割、物体跟踪、行为分析等其他高层视觉任务的基础，在许多领域都有着广泛的应用。如：安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

在图像分类任务中，我们向大家介绍如何训练AlexNet、VGG、GoogLeNet、ResNet、Inception-v4、Inception-Resnet-V2和Xception模型。同时提供了能够将Caffe或TensorFlow训练好的模型文件转换为PaddlePaddle模型文件的模型转换工具。

- 11.1 [将Caffe模型文件转换为PaddlePaddle模型文件](https://github.com/PaddlePaddle/models/tree/develop/image_classification/caffe2paddle)
- 11.2 [将TensorFlow模型文件转换为PaddlePaddle模型文件](https://github.com/PaddlePaddle/models/tree/develop/image_classification/tf2paddle)
- 11.3 [AlexNet](https://github.com/PaddlePaddle/models/tree/develop/image_classification)
- 11.4 [VGG](https://github.com/PaddlePaddle/models/tree/develop/image_classification)
- 11.5 [Residual Network](https://github.com/PaddlePaddle/models/tree/develop/image_classification)
- 11.6 [Inception-v4](https://github.com/PaddlePaddle/models/tree/develop/image_classification)
- 11.7 [Inception-Resnet-V2](https://github.com/PaddlePaddle/models/tree/develop/image_classification)
- 11.8 [Xception](https://github.com/PaddlePaddle/models/tree/develop/image_classification)

## 12. 目标检测

目标检测任务的目标是给定一张图像或是视频帧，让计算机找出其中所有目标的位置，并给出每个目标的具体类别。对于人类来说，目标检测是一个非常简单的任务。然而，计算机能够“看到”的仅有一些值为0 ~ 255的矩阵，很难解图像或是视频帧中出现了人或是物体这样的高层语义概念，也就更加难以定位目标出现在图像中哪个区域。与此同时，由于目标会出现在图像或是视频帧中的任何位置，目标的形态千变万化，图像或是视频帧的背景千差万别，诸多因素都使得目标检测对计算机来说是一个具有挑战性的问题。

在目标检测任务中，我们介绍利用SSD方法完成目标检测。SSD全称：Single Shot MultiBox Detector，是目标检测领域较新且效果较好的检测算法之一，具有检测速度快且检测精度高的特点。

- 12.1 [Single Shot MultiBox Detector](https://github.com/PaddlePaddle/models/tree/develop/ssd/README.cn.md)

## 13. 场景文字识别

许多场景图像中包含着丰富的文本信息，对理解图像信息有着重要作用，能够极大地帮助人们认知和理解场景图像的内容。场景文字识别是在图像背景复杂、分辨率低下、字体多样、分布随意等情况下，将图像信息转化为文字序列的过程，可认为是一种特别的翻译过程：将图像输入翻译为自然语言输出。场景图像文字识别技术的发展也促进了一些新型应用的产生，如通过自动识别路牌中的文字帮助街景应用获取更加准确的地址信息等。

在场景文字识别任务中，我们介绍如何将基于CNN的图像特征提取和基于RNN的序列翻译技术结合，免除人工定义特征，避免字符分割，使用自动学习到的图像特征，完成端到端地无约束字符定位和识别。

- 13.1 [场景文字识别](https://github.com/PaddlePaddle/models/tree/develop/scene_text_recognition)

## 14. 语音识别

语音识别技术（Auto Speech Recognize，简称ASR）将人类语音中的词汇内容转化为计算机可读的输入，让机器能够“听懂”人类的语音，在语音助手、语音输入、语音交互等应用中发挥着重要作用。深度学习在语音识别领域取得了瞩目的成绩，端到端的深度学习方法将传统的声学模型、词典、语言模型等模块融为一个整体，不再依赖隐马尔可夫模型中的各种条件独立性假设，令模型变得更加简洁，一个神经网络模型以语音特征为输入，直接输出识别出的文本，目前已经成为语音识别最重要的手段。

在语音识别任务中，我们提供了基于 DeepSpeech2 模型的完整流水线，包括：特征提取、数据增强、模型训练、语言模型、解码模块等，并提供一个训练好的模型和体验实例，大家能够使用自己的声音来体验语音识别的乐趣。

14.1 [语音识别: DeepSpeech2](https://github.com/PaddlePaddle/DeepSpeech)

本教程由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)创作，采用[Apache-2.0](LICENSE) 许可协议进行许可。
