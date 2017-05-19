# NCE加速词向量训练
## 背景介绍
神经概率语言模型 (NPLM) 尽管优异的性能，但是其使用率仍然远远低于n-gram传统模型的使用，这是由于其众所周知的漫长训练时间。训练 NPLM 计算开销很大，因为softmax需要全局的计算，必须考虑词汇中的所有单词。
NCE（Noise-contrastive estimation）是一种快速简便的训练 NPLM 的算法，一种新的连续分布估计方法。这里我们使用了ptb数据来训练神经语言模型，并表明它减少了训练时间超过一个数量级, 而不影响所产生的模型的质量。该算法比重要性采样更有效而且更稳定, 因为它需要的噪音样本要少得多\[[1](#参考文献)\]。

## 实验数据
本文采用Penn Treebank (PTB)数据集（[Tomas Mikolov预处理版本](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)），这是一个用来训练语言模型的数据集，给出前4个词让语言模型预测第5个词。PaddlePaddle提供[paddle.dataset.imikolov](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/dataset/imikolov.py)接口来方便调用数据，其中实现了数据自动下载，字典生成，滑动窗口等功能。

## NCE Layer
NCE Layer引用自论文\[[1](#参考文献)\]，NCE是指Noise-contrastive estimation，目的是用来提高训练速度并改善所得词向量的质量。与h-sigmoid[[2](#参考文献)\]相比，NCE不再使用复杂的Huffman树来构造目标函数，而是采用相对简单的随机负采样，这样能够大幅度提升计算性能。

NCE原理是通过构造一个逻辑回归（logistic regression），对正样例和负样例做二分类，对于每一个样本，将自身的预测词label作为正样例，同时采样出K个其他词label作为负样例，从而只需要计算样本在这K+1个label上的概率。整体目标函数的目的就是增大正样本的概率同时降低负样本的概率。相比原始的softmax分类需要计算每个类别的分数，然后归一化得到概率，这个计算过程是十分耗时的。

## 网络结构
**模型的总体结构：**

本文通过训练N-gram语言模型来获得词向量，具体地使用前4个词来预测当前词。网络输入为词在字典中的id，然后查询词向量词表获取词向量，接着拼接4个词的词向量，然后接入一个全连接隐层，最后是NCE Layer层。详细网络结构见图1：

<p align="center">
<img src="images/network_conf.png" width = "80%" align="center"/><br/>
图1. 网络配置结构
</p>
**可以看到，模型主要分为如下几个部分：**

- **输入层**：ptb的样本由原始的英文单词组成，将每个英文单词转换为字典中的id表示。

- **词向量层**：使用定义好的embedding矩阵，将原先的id表示转换为向量表示。这种将英文单词转换为词向量的方法，比传统的one-hot表示更能体现词语的语义内容，关于词向量的更多信息请参考PaddleBook中的[词向量](https://github.com/PaddlePaddle/book/tree/develop/04.word2vec)一节。

- **词向量拼接层**：将词向量进行并联，就是将向量沿feature边依次拼接在一起形成一个矩阵。

- **全连接隐层**：将上一层获得的词向量矩阵输入一层隐层的神经网络，

- **NCE层**：推断时，输出层的神经元数量和样本的类别数一致，在这里就是整个字典的大小，最后使用softmax对每个类别的概率做一个归一化操作，因此第$i$个神经元的输出就可以认为是样本属于第$i$类的预测概率。训练时，我们需要构造一个二分类分类器，万幸的是paddle已经帮助我们实现了这一切。

## 训练阶段
训练直接运行``` python train.py ```。程序第一次运行会检测用户缓存文件夹中是否包含imikolov数据集，如果未包含，则自动下载。运行过程中，每1000个iteration会打印模型训练信息，主要包含训练损失，每个pass计算一次测试损失，并会保存一次模型。

在PaddlePaddle中也有已经实现好的nce layer，这里有一些参数需要自行根据实际场景进行设计，例如param\_attr和bias\_attr这两个参数，这是用来设置参数名字，这是为了后面预测阶段好来实现网络的参数共享，具体内容下一个章节里会称述。num\_neg\_samples参数负责控制对负样例的采样个数，同时也是相对正样例的采样倍数。neg\_distribution可以控制生成负样例标签的分布，默认是一个均匀分布。act参数表示激活函数，根据NCE的原理，这里应该使用sigmoid函数。

## 预测阶段
预测直接运行``` python infer.py ```，程序会首先load最新模型，然后按照batch方式进行预测，并打印预测结果。预测阶段最重要的就是共享NCE layer中的逻辑回归训练得到的参数，因为PaddlePaddle里的NCE层并不支持直接用在预测时进行输出，所以必须自己重新写一个推断层，推断层的参数为训练时的参数，所以需要参数共享。

参数分享的方法，通过paddle.attr.Param方法获取参数值，并参数值传入paddle.layer.trans\_full\_matrix\_projection对隐层输出向量hidden\_layer做一个矩阵右乘，从而得到最后的类别向量，将类别向量输入softmax做一个归一操作，从而得到最后的类别概率分布。

代码实现如下：

```python
if is_train == True:
    cost = paddle.layer.nce(
        input=hidden_layer,
        label=next_word,
        num_classes=dict_size,
        param_attr=paddle.attr.Param(name='nce_w'),
        bias_attr=paddle.attr.Param(name='nce_b'),
        act=paddle.activation.Sigmoid(),
        num_neg_samples=25,
        neg_distribution=None)
    return cost
else:
    with paddle.layer.mixed(
            size=dict_size,
            act=paddle.activation.Softmax(),
            bias_attr=paddle.attr.Param(name='nce_b')) as prediction:
        prediction += paddle.layer.trans_full_matrix_projection(
            input=hidden_layer, param_attr=paddle.attr.Param(name='nce_w'))
```

## 参考文献
1. Mnih A, Kavukcuoglu K. [Learning word embeddings efficiently with noise-contrastive estimation](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)[C]//Advances in neural information processing systems. 2013: 2265-2273.

2. Morin, F., & Bengio, Y. (2005, January). [Hierarchical Probabilistic Neural Network Language Model](http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf). In Aistats (Vol. 5, pp. 246-252).
