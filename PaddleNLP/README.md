PaddleNLP
=========

机器翻译
--------

机器翻译（Machine Translation）将一种自然语言(源语言)转换成一种自然语言（目标语言），是自然语言处理中非常基础和重要的研究方向。在全球化的浪潮中，机器翻译在促进跨语言文明的交流中所起的重要作用是不言而喻的。其发展经历了统计机器翻译和基于神经网络的神经机器翻译(Nueural
Machine Translation, NMT)等阶段。在 NMT 成熟后，机器翻译才真正得以大规模应用。而早阶段的 NMT 主要是基于循环神经网络 RNN 的，其训练过程中当前时间步依赖于前一个时间步的计算，时间步之间难以并行化以提高训练速度。因此，非 RNN 结构的 NMT 得以应运而生，例如基 卷积神经网络 CNN 的结构和基于自注意力机制（Self-Attention）的结构。

本实例所实现的 Transformer 就是一个基于自注意力机制的机器翻译模型，其中不再有RNN或CNN结构，而是完全利用 Attention 学习语言中的上下文依赖。相较于RNN/CNN, 这种结构在单层内计算复杂度更低、易于并行化、对长程依赖更易建模，最终在多种语言之间取得了最好的翻译效果。

-  [Transformer](https://github.com/PaddlePaddle/models/blob/release/1.4/PaddleNLP/neural_machine_translation/transformer/README_cn.md)


中文词法分析
------------

中文分词(Word Segmentation)是将连续的自然语言文本，切分出具有语义合理性和完整性的词汇序列的过程。因为在汉语中，词是承担语义的最基本单位，切词是文本分类、情感分析、信息检索等众多自然语言处理任务的基础。 词性标注（Part-of-speech Tagging）是为自然语言文本中的每一个词汇赋予一个词性的过程，这里的词性包括名词、动词、形容词、副词等等。 命名实体识别（Named Entity Recognition，NER）又称作“专名识别”，是指识别自然语言文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等。 我们将这三个任务统一成一个联合任务，称为词法分析任务，基于深度神经网络，利用海量标注语料进行训练，提供了一个端到端的解决方案。

我们把这个联合的中文词法分析解决方案命名为LAC。LAC既可以认为是Lexical Analysis of Chinese的首字母缩写，也可以认为是LAC Analyzes Chinese的递归缩写。

- [LAC](https://github.com/baidu/lac/blob/master/README.md)

情感倾向分析
------------

情感倾向分析针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极、中性。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有力的决策支持。本次我们开放 AI 开放平台中情感倾向分析采用的[模型](http://ai.baidu.com/tech/nlp/sentiment_classify)，提供给用户使用。

- [Senta](https://github.com/baidu/Senta/blob/master/README.md)

语义匹配
--------

在自然语言处理很多场景中，需要度量两个文本在语义上的相似度，这类任务通常被称为语义匹配。例如在搜索中根据查询与候选文档的相似度对搜索结果进行排序，文本去重中文本与文本相似度的计算，自动问答中候选答案与问题的匹配等。

本例所开放的DAM (Deep Attention Matching Network)为百度自然语言处理部发表于ACL-2018的工作，用于检索式聊天机器人多轮对话中应答的选择。DAM受Transformer的启发，其网络结构完全基于注意力(attention)机制，利用栈式的self-attention结构分别学习不同粒度下应答和语境的语义表示，然后利用cross-attention获取应答与语境之间的相关性，在两个大规模多轮对话数据集上的表现均好于其它模型。

- [Deep Attention Matching Network](https://github.com/PaddlePaddle/models/tree/release/1.4/PaddleNLP/deep_attention_matching_net)

AnyQ
----

[AnyQ](https://github.com/baidu/AnyQ)(ANswer Your Questions) 开源项目主要包含面向FAQ集合的问答系统框架、文本语义匹配工具SimNet。 问答系统框架采用了配置化、插件化的设计，各功能均通过插件形式加入，当前共开放了20+种插件。开发者可以使用AnyQ系统快速构建和定制适用于特定业务场景的FAQ问答系统，并加速迭代和升级。

SimNet是百度自然语言处理部于2013年自主研发的语义匹配框架，该框架在百度各产品上广泛应用，主要包括BOW、CNN、RNN、MM-DNN等核心网络结构形式，同时基于该框架也集成了学术界主流的语义匹配模型，如MatchPyramid、MV-LSTM、K-NRM等模型。使用SimNet构建出的模型可以便捷的加入AnyQ系统中，增强AnyQ系统的语义匹配能力。

-  [SimNet in PaddlePaddle Fluid](https://github.com/baidu/AnyQ/blob/master/tools/simnet/train/paddle/README.md)

机器阅读理解
----------

机器阅读理解(MRC)是自然语言处理(NLP)中的核心任务之一，最终目标是让机器像人类一样阅读文本，提炼文本信息并回答相关问题。深度学习近年来在NLP中得到广泛使用，也使得机器阅读理解能力在近年有了大幅提高，但是目前研究的机器阅读理解都采用人工构造的数据集，以及回答一些相对简单的问题，和人类处理的数据还有明显差距，因此亟需大规模真实训练数据推动MRC的进一步发展。

百度阅读理解数据集是由百度自然语言处理部开源的一个真实世界数据集，所有的问题、原文都来源于实际数据(百度搜索引擎数据和百度知道问答社区)，答案是由人类回答的。每个问题都对应多个答案，数据集包含200k问题、1000k原文和420k答案，是目前最大的中文MRC数据集。百度同时开源了对应的阅读理解模型，称为DuReader，采用当前通用的网络分层结构，通过双向attention机制捕捉问题和原文之间的交互关系，生成query-aware的原文表示，最终基于query-aware的原文表示通过point network预测答案范围。

-  [DuReader in PaddlePaddle Fluid](https://github.com/PaddlePaddle/models/blob/release/1.4/PaddleNLP/machine_reading_comprehension/README.md)
