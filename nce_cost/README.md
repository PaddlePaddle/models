# NCE加速词向量训练
## 背景介绍
在自然语言处理领域中，传统做法通常使用one-hot向量来表示词，比如词典为['我', '你', '喜欢']，可以用[1,0,0]、[0,1,0]和[0,0,1]这三个向量分别表示'我'、'你'和'喜欢'。这种表示方式比较简洁，但是当词表很大时，容易产生维度爆炸问题；而且任意两个词的向量是正交的，向量包含的信息有限。为了避免或减轻one-hot表示的缺点，目前通常使用词向量来取代one-hot表示，词向量也就是word embedding，即使用一个低维稠密的实向量取代高维稀疏的one-hot向量。训练词向量的方法有很多种，神经网络模型是其中之一，包括CBOW、Skip-gram等，这些模型本质上都是一个分类模型，当词表较大即类别较多时，传统的softmax将非常消耗时间。PaddlePaddle提供了Hsigmoid Layer、NCE Layer，来加速模型的训练过程。本文主要介绍如何使用NCE Layer来加速训练，词向量相关内容请查阅PaddlePaddle Book中的[词向量章节](https://github.com/PaddlePaddle/book/tree/develop/04.word2vec)。

## NCE Layer
NCE Layer引用自论文\[[1](#参考文献)\]，NCE是指Noise-contrastive estimation，原理是通过构造一个逻辑回归（logistic regression），对正样例和负样例做二分类，对于每一个样本，将自身的label作为正样例，同时采样出K个其他的label作为负样例，从而只需要计算样本在这K+1个label上的概率，原始的softmax分类需要计算每个类别的分数，然后归一化得到概率，这个计算过程是十分耗时的。
## 数据准备
### PTB数据
本文采用Penn Treebank (PTB)数据集（[Tomas Mikolov预处理版本](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)），共包含train、valid和test三个文件。其中使用train作为训练数据，valid作为测试数据。本文训练的是5-gram模型，即用每条数据的前4个词来预测第5个词。PaddlePaddle提供了对应PTB数据集的python包[paddle.dataset.imikolov](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/dataset/imikolov.py)    ，自动做数据的下载与预处理。预处理会把数据集中的每一句话前后加上开始符号\<s>以及结束符号\<e>，然后依据窗口大小（本文为5），从头到尾每次向右滑动窗口并生成一条数据。如"I have a dream that one day"可以生成\<s> I have a dream、I have a dream that、have a dream that one、a dream that one day、dream that one day \<e>，PaddlePaddle会把词转换成id数据作为预处理的输出。

## 网络结构
本文通过训练N-gram语言模型来获得词向量，具体地使用前4个词来预测当前词。网络输入为词在字典中的id，然后查询词向量词表获取词向量，接着拼接4个词的词向量，然后接入一个全连接隐层，最后是NCE Layer层。详细网络结构见图1：

<p align="center">
<img src="images/network_conf.png" width = "70%" align="center"/><br/>
图1. 网络配置结构
</p>

## 训练阶段
训练直接运行``` python train.py ```。程序第一次运行会检测用户缓存文件夹中是否包含imikolov数据集，如果未包含，则自动下载。运行过程中，每1000个iteration会打印模型训练信息，主要包含训练损失，每个pass计算一次测试损失，并会保存一次模型。

## 预测阶段
预测直接运行``` python infer.py ```，程序会首先load最新模型，然后按照batch方式进行预测，并打印预测结果。预测阶段最重要的就是共享NCE layer中的逻辑回归训练得到的参数。

## 参考文献
1. Mnih A, Kavukcuoglu K. [Learning word embeddings efficiently with noise-contrastive estimation](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)[C]//Advances in neural information processing systems. 2013: 2265-2273.
