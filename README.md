# models 简介

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/models)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://github.com/PaddlePaddle/models)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle提供了丰富的运算单元，帮助大家以模块化的方式构建起千变万化的深度学习模型来解决不同的应用问题。这里，我们针对常见的机器学习任务，提供了不同的神经网络模型供大家学习和使用。

## [词向量](https://github.com/PaddlePaddle/models/tree/develop/word_embedding)

- **介绍**

    [词向量](https://github.com/PaddlePaddle/book/blob/develop/04.word2vec/README.cn.md) 是深度学习应用于自然语言处理领域最成功的概念和成果之一，是一种分散式表示（distributed representation）法。分散式表示法用一个更低维度的实向量表示词语，向量的每个维度在实数域取值，都表示文本的某种潜在语法或语义特征。广义地讲，词向量也可以应用于普通离散特征。词向量的学习通常都是一个无监督的学习过程，因此，可以充分利用海量的无标记数据以捕获特征之间的关系，也可以有效地解决特征稀疏、标签数据缺失、数据噪声等问题。

    然而，在常见词向量学习方法中，模型最后一层往往会遇到一个超大规模的分类问题，是计算性能的瓶颈。在词向量的例子中，我们向大家展示如何使用Hierarchical-Sigmoid 和噪声对比估计（Noise Contrastive Estimation，NCE）来加速词向量的学习。

- **应用领域**

    词向量是深度学习方法引入自然语言处理领域的核心技术之一，在大规模无标记语料上训练的词向量常作为各种自然语言处理任务的预训练参数，是一种较为通用的资源，对任务性能的进一步提升有一定的帮助。同时，词嵌入的思想也是深度学习模型处理离散特征的重要方法，有着广泛地借鉴和参考意义。

    词向量是搜索引擎、广告系统、推荐系统等互联网服务背后的常见基础技术之一。

- **模型配置**

    1. [Hsigmoid加速词向量训练](https://github.com/PaddlePaddle/models/blob/develop/word_embedding/hsigmoid_train.py)
    2. [噪声对比估计加速词向量训练]()

## [文本分类](https://github.com/PaddlePaddle/models/tree/develop/text_classification)

- **介绍**

    文本分类是自然语言处理领域最基础的任务之一，深度学习方法能够免除复杂的特征工程，直接使用原始文本作为输入，数据驱动地最优化分类准确率。我们以情感分类任务为例，提供了基于DNN的非序列文本分类模型，基于CNN和LSTM的序列模型供大家学习和使用。

- **应用领域**

    分类是机器学习基础任务之一。文本分类模型在SPAM检测，文本打标签，文本意图识别，文章质量评估，色情暴力文章识别，评论情绪识别，广告物料风险控制等领域都有着广泛的应用。

- **模型配置**

    1. [基于 DNN 的文本分类](https://github.com/PaddlePaddle/models/blob/develop/text_classification/text_classification_dnn.py)
    2. [基于 CNN 的文本分类](https://github.com/PaddlePaddle/models/blob/develop/text_classification/text_classification_cnn.py)
    3. [基于 LSTM 的文本分类](https://github.com/PaddlePaddle/book/blob/develop/06.understand_sentiment/train.py)

## [序列标注](https://github.com/PaddlePaddle/models/tree/develop/sequence_tagging_for_ner)

- **介绍**

    序列标注是自然语言处理中最常见的问题之一。在这一任务中，给定输入序列，模型为序列中每一个元素贴上一个类别标签。随着深度学习的不断探索和发展，利用循环神经网络模型学习输入序列的特征表示，条件随机场（Conditional Random Field, CRF）在特征基础上完成序列标注任务，逐渐成为解决序列标注问题的标配解决方案。深度学习的巨大优势在于：从原始文本中学习，避免复杂的特征工程，只要构建好这样一套深度学习模型，绝大多数序列标注问题都可以直接套用，只需要相应地替换不同问题对应的训练数据。

    这里，我们以命名实体识别（Named Entity Recognition，NER）任务为例，向大家介绍如何使用 PaddlePaddle 训练一个端到端（End-to-End）的序列标注模型。

- **应用领域**

    序列标注是自然语言处理领域最重要的基础任务之一，有着广泛地应用：自动分词，语言角色标注，命名实体识别，深度问答（Deep QA），关键词提取等问题都可以转化为序列标注问题直接套用本例中的模型。

- **模型配置**

    1. [命名实体识别](https://github.com/PaddlePaddle/models/blob/develop/sequence_tagging_for_ner/ner.py)

## [排序学习](https://github.com/PaddlePaddle/models/tree/develop/ltr)

- **介绍**

    排序学习（Learning to Rank，下简称LTR）是信息检索和搜索引擎研究的核心问题之一，通过机器学习方法学习一个分值函数（Scoring Function）对待排序的候选进行打分，再根据分值的高低确定序关系。深度神经网络可以用来建模分值函数，构成各类基于深度学习的LTR模型。

    以信息检索任务为例，给定查询以及检索到的候选文档列表，LTR系统需要按照查询与候选文档的相关性，对候选文档进行打分并排序。流行的LTR学习方法分为以下三种：

    - [Pointwise](https://github.com/PaddlePaddle/book/blob/develop/05.recommender_system/README.cn.md)
        - Pointwise 学习方法将LTR被转化为回归或是分类问题。
        - 给定查询以及一个候选文档，模型基于序数进行二分类、多分类或者回归拟合，是一种基础的LTR学习策略。
    - Pairwise
        - Pairwise学习方法将排序问题归约为对有序对（ordered pair）的分类，比Pointwise方法更近了一步。
        - 模型判断一对候选文档中，哪一个与给定查询更相关，学习的目标为是最小化误分类文档对的数量。
        - 理想情况下，如果所有文档对都能被正确的分类，那么原始的候选文档也会被正确的排序。
    - Listwise
        - 与Pointwise与Pairwise 学习方法相比，Listwise方法将给定查询对应的整个候选文档集合列表（list）作为输入，直接对排序结果列表进行优化。Listwise方法在损失函数中考虑了文档排序的位置因素，是前两种方法所不具备的。

    Pointwise 学习策略是PaddleBook中[推荐系统](https://github.com/PaddlePaddle/book/blob/develop/05.recommender_system/README.cn.md)一节介绍过的方法。这里，我们一进步提供了基于 RankLoss 损失函数的 Pairwise 排序模型和基于LambdaRank损失函数的Listwise排序模型。

- **应用领域**

    LTR模型在搜索排序，包括：图片搜索排序、外卖美食搜索排序、App搜索排序、酒店搜索排序等场景中有着广泛的应用，还可以扩展应用于：关键词推荐、各类业务榜单、个性化推荐等任务。

- **模型配置说明**

    1. [Pointwise 排序模型](https://github.com/PaddlePaddle/book/blob/develop/05.recommender_system/train.py)
    2. [Pairwise 排序模型](https://github.com/PaddlePaddle/models/blob/develop/ltr/ranknet.py)
    3. [Listwise 排序模型](https://github.com/PaddlePaddle/models/blob/develop/ltr/lambda_rank.py)

## [文本生成](https://github.com/PaddlePaddle/models/tree/develop/nmt_without_attention)

- **介绍**

    我们期待有一天机器可以使用自然语言与人们进行交流，像人一样能够撰写高质量的自然语言文本，自动文本生成是实现这一目标的关键技术，可以应用于机器翻译系统、对话系统、问答系统等，为人们带来更加有趣地交互体验，也可以自动撰写新闻摘要，撰写歌词，简单的故事等等。或许未来的某一天，机器能够代替编辑，作家，歌词作者，颠覆这些内容创作领域的工作方式。

    基于神经网络生成文本常使用两类方法：1. 语言模型；2. 序列到序列（sequence to sequence）映射模型。在文本生成的例子中，我们为大家展示如何使用以上两种模型来自动生成文本。

    特别的，对序列到序列映射模型，我们以机器翻译任务为例提供了多种改进模型，供大家学习和使用，包括：
    1. 不带注意力机制的序列到序列映射模型，这一模型是所有序列到序列学习模型的基础。
    2. 带注意力机制使用 scheduled sampling 改善生成质量，用来改善RNN模型在文本生成过程中的错误累积问题。
    3. 带外部记忆机制的神经机器翻译，通过增强神经网络的记忆能力，来完成复杂的序列到序列学习任务。


- **应用领域**

    文本生成模型实现了两个甚至是多个不定长模型之间的映射，有着广泛地应用，包括机器翻译、智能对话与问答、广告创意语料生成、自动编码（如金融画像编码）、判断多个文本串之间的语义相关性等。

- **模型配置**

    1. [无注意力机制的编码器解码器模型](https://github.com/PaddlePaddle/models/blob/develop/nmt_without_attention/nmt_without_attention.py)
    2. [使用 scheduled sampling 改善生成质量]()
    3. [ 带外部记忆机制的神经机器翻译]()

## Copyright and License
PaddlePaddle is provided under the [Apache-2.0 license](LICENSE).
