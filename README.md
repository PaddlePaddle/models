# models 简介

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/models)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://github.com/PaddlePaddle/models)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddlePaddle提供了丰富的运算单元，帮助大家以模块化的方式构建起千变万化的深度学习模型来解决不同的应用问题。这里，我们针对常见的机器学习任务，提供了不同的神经网络模型供大家学习和使用。


## 1. 词向量

词向量用一个实向量表示词语，向量的每个维都表示文本的某种潜在语法或语义特征，是深度学习应用于自然语言处理领域最成功的概念和成果之一。广义地，词向量也可以应用于普通离散特征。词向量的学习通常都是一个无监督的学习过程，因此，可以充分利用海量的无标记数据以捕获特征之间的关系，也可以有效地解决特征稀疏、标签数据缺失、数据噪声等问题。然而，在常见词向量学习方法中，模型最后一层往往会遇到一个超大规模的分类问题，是计算性能的瓶颈。

在词向量的例子中，我们向大家展示如何使用Hierarchical-Sigmoid 和噪声对比估计（Noise Contrastive Estimation，NCE）来加速词向量的学习。

- 1.1 [Hsigmoid加速词向量训练](https://github.com/PaddlePaddle/models/tree/develop/word_embedding)

## 2. 点击率预估

点击率预估模型预判用户对一条广告点击的概率，对每次广告的点击情况做出预测，是广告技术的核心算法之一。逻谛斯克回归对大规模稀疏特征有着很好的学习能力，在发展点击率预估任务的早期一统天下。近年来，DNN 模型由于其强大的学习能力逐渐接过点击率预估任务的大旗。

在点击率预估的例子中，我们给出谷歌提出的 Wide & Deep 模型。这一模型融合了适用于学习抽象特征的 DNN 和适用于大规模稀疏特征的逻谛斯克回归两者模型的优点，可以作为一种相对成熟的模型框架使用， 在工业界也有一定的应用。

- 2.1 [Wide & deep 点击率预估模型](https://github.com/PaddlePaddle/models/tree/develop/ctr)

## 3. 文本分类

文本分类是自然语言处理领域最基础的任务之一，深度学习方法能够免除复杂的特征工程，直接使用原始文本作为输入，数据驱动地最优化分类准确率。

在文本分类的例子中，我们以情感分类任务为例，提供了基于DNN的非序列文本分类模型，以及基于CNN的序列模型供大家学习和使用（基于LSTM的模型见PaddleBook中[情感分类](https://github.com/PaddlePaddle/book/blob/develop/06.understand_sentiment/README.cn.md)一课）。

- 3.1 [基于 DNN / CNN 的情感分类](https://github.com/PaddlePaddle/models/tree/develop/text_classification)

## 4. 排序学习

排序学习是信息检索和搜索引擎研究的核心问题之一，通过机器学习方法学习一个分值函数对待排序的候选进行打分，再根据分值的高低确定序关系。深度神经网络可以用来建模分值函数，构成各类基于深度学习的LTR模型。

在排序学习的例子中，我们介绍基于 RankLoss 损失函数的 Pairwise 排序模型和基于LambdaRank损失函数的Listwise排序模型(Pointwise学习策略见PaddleBook中[推荐系统](https://github.com/PaddlePaddle/book/blob/develop/05.recommender_system/README.cn.md)一课）。

- 4.1 [基于 Pointwise 的排序学习](https://github.com/PaddlePaddle/book/blob/develop/05.recommender_system/README.cn.md)
- 4.2 [基于 Pairwise 和 Listwise 的排序学习](https://github.com/PaddlePaddle/models/tree/develop/ltr)

## 5. 序列标注

序列标注是自然语言处理中最常见的问题之一。在这一任务中，给定输入序列，模型为序列中每一个元素贴上一个类别标签。随着深度学习的不断探索和发展，利用循环神经网络模型学习输入序列的特征表示，条件随机场（Conditional Random Field, CRF）在特征基础上完成序列标注任务，逐渐成为解决序列标注问题的标配解决方案。

在序列标注的例子中，我们以命名实体识别（Named Entity Recognition，NER）任务为例，介绍如何训练一个端到端的序列标注模型。

- 5.1 [命名实体识别](https://github.com/PaddlePaddle/models/tree/develop/sequence_tagging_for_ner)

## 6. 序列到序列学习

序列到序列学习实现两个甚至是多个不定长模型之间的映射，有着广泛地应用，包括：机器翻译、智能对话与问答、广告创意语料生成、自动编码（如金融画像编码）、判断多个文本串之间的语义相关性等。

在序列到序列学习的例子中，我们以机器翻译任务为例，提供了多种改进模型，供大家学习和使用。包括：不带注意力机制的序列到序列映射模型，这一模型是所有序列到序列学习模型的基础；使用 scheduled sampling 改善 RNN 模型在生成任务中的错误累积问题；带外部记忆机制的神经机器翻译，通过增强神经网络的记忆能力，来完成复杂的序列到序列学习任务。

- 6.1 [无注意力机制的编码器解码器模型](https://github.com/PaddlePaddle/models/tree/develop/nmt_without_attention)

## Copyright and License
PaddlePaddle is provided under the [Apache-2.0 license](LICENSE).
