PaddleRec
=========

个性化推荐
-------

推荐系统在当前的互联网服务中正在发挥越来越大的作用，目前大部分电子商务系统、社交网络，广告推荐，搜索引擎，都不同程度的使用了各种形式的个性化推荐技术，帮助用户快速找到他们想要的信息。

在工业可用的推荐系统中，推荐策略一般会被划分为多个模块串联执行。以新闻推荐系统为例，存在多个可以使用深度学习技术的环节，例如新闻的自动化标注，个性化新闻召回，个性化匹配与排序等。PaddlePaddle对推荐算法的训练提供了完整的支持，并提供了多种模型配置供用户选择。

- [TagSpace](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/tagspace)
- [GRU4Rec](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gru4rec)
- [SequenceSemanticRetrieval](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ssr)
- [DeepCTR](https://github.com/PaddlePaddle/models/blob/develop/PaddleRec/ctr/README.cn.md)
- [Multiview-Simnet](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/multiview_simnet)
- [Word2Vec](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/word2vec)
- [GraphNeuralNetwork](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gnn)
- [DeepInterestNetwork](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/din)


|                        模型                        |                           应用场景                           |                             简介                             |
| :------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [GRU4Rec](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gru4rec) | session-based推荐 | 首次将RNN（GRU）运用于session-based推荐，核心思想是在一个session中，用户点击一系列item的行为看做一个序列，用来训练RNN模型 |
|               **词向量（word2vec）**               |                         [word2vec]()                         | 提供单机多卡，多机等分布式训练中文词向量能力，支持主流>词向量模型（skip-gram，cbow等），可以快速使用自定义数据训练词向量模型。 |
|                    **语言模型**                    | [Language_model](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/language_model) | 基于循环神经网络（RNN）的经典神经语言模型（neural language model）。 |
|                 **情感分类**:fire:                 | [Senta](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/sentiment_classification)，[EmotionDetection](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/emotion_detection) | Senta（Sentiment Classification，简称Senta）和EmotionDetection两个项目分别提供了面向*通用场 >景*和*人机对话场景专用*的情感倾向性分析模型。 |
|              **文本相似度计算**:fire:              | [SimNet](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/similarity_net) | SimNet，又称为Similarity Net>，为您提供高效可靠的文本相似度计算工具和预训练模型。 |
