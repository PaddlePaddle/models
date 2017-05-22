# 噪声对比估计加速词向量训练
## 背景介绍
语言模型是自然语言处理领域的基础问题，其在词性标注、句法分析、机器翻译、信息检索等任务中起到了重要作用。

神经概率语言模型（Neural Probabilistic Language Model, NPLM）尽管有优异的精度表现，但是相对基于统计的 n-gram 传统模型，训练时间还是太漫长了[[4](#参考文献)]。原因请见[下一章节](#NCE Layer)。

NCE（Noise-contrastive estimation）[[2](#参考文献)]，是一种快速简便的离散分布估计方法，这里以训练 NPLM 为例。这里我们使用了 ptb 数据来训练神经语言模型。

## NCE Layer
在这里将 NCE 用于训练神经语言模型，主要目的是用来提高训练速度。训练 NPLM 计算开销很大，是因为 softmax 函数计算时需要计算每个类别的指数项，必须考虑字典中的所有单词，这是相当耗时的，因为对于语言模型字典往往非常大[[4](#参考文献)]。与 hierarchical-sigmoid \[[3](#参考文献)\] 相比，NCE 不再使用复杂的 Huffman 树来构造目标函数，而是采用相对简单的随机负采样，以大幅提升计算效率。


假设已知具体的上下文 h，并且知道这个分布为 ${ P }^{ h }(w)$ ，我们将训练样例作为正样例，从一个噪音分布 ${ P }_{ n }(w)$ 抽样产生负样例。我们可以任意选择合适的噪音分布，默认为无偏的均匀分布。这里我们同时假设噪音样例 k 倍于数据样例，则训练数据被抽中的概率为[[2](#参考文献)]：

$$
{ P }^{ h }(D=1|w,\theta )=\frac { { P }_{ \theta  }^{ h }(w) }{ { P }^{ h }_{ \theta  }(w)+k{ P }_{ n }(w) } =\sigma (\Delta { s }_{ \theta  }^{  }(w,h))
$$

其中 $\Delta { s }_{ \theta  }^{  }(w,h)={ s }_{ \theta  }^{  }(w,h)-\log { (k{ P }_{ n }^{  }(w)) }$ ，${ s }_{ \theta  }(w,h)$ 表示选择在生成 $w$ 字并处于上下文 $h$ 时的特征向量，整体目标函数的目的就是增大正样本的概率同时降低负样本的概率。目标函数如下[[2](#参考文献)]：

$$
{ J }_{  }^{ h }(\theta )={ E  }_{ { P }_{ d }^{ h } }^{  }\left[ \log { { P }_{  }^{ h }(D=1|w,\theta ) }  \right] +k{ E }_{ { P }_{ n }^{  } }^{  }\left[ \log { { P }_{  }^{ h } } (D=0|w,\theta ) \right] \\ \qquad ={ E }_{ { P }_{ d }^{ h } }^{  }\left[ \log { \sigma (\Delta { s }_{ \theta  }^{  }(w,h)) }  \right] +k{ E  }_{ { P }_{ n }^{  } }^{  }\left[ \log { (1-\sigma (\Delta { s }_{ \theta  }^{  }(w,h))) }  \right]
$$

NCE 原理是通过构造一个逻辑回归（logistic regression），对正样例和负样例做二分类，对于每一个样本，将自身的预测词 label 作为正样例，同时采样出 k 个其他词 label 作为负样例，从而只需要计算样本在这 k+1 个 label 上的概率。相比原始的 softmax 分类需要计算每个类别的分数，然后归一化得到概率，softmax 这个计算过程是十分耗时的。

## 实验数据
本文采用 Penn Treebank (PTB)数据集（[Tomas Mikolov预处理版本](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)），这是一个可以用来训练语言模型的数据集。PaddlePaddle 提供 [paddle.dataset.imikolov](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/dataset/imikolov.py) 接口来方便调用数据，其中实现了数据自动下载，字典生成，滑动窗口等功能。数据接口给出的是前4个词让语言模型预测第5个词。

## 网络结构
本文在训练 n-gram 语言模型时，即使用前4个词来预测当前词。网络输入为词在字典中的 id，然后查询词向量词表获取词向量，接着拼接4个词的词向量，然后接入一个全连接隐层，最后是 NCE 层。详细网络结构见图1：

<p align="center">
<img src="images/network_conf.png" width = "80%" align="center"/><br/>
图1. 网络配置结构
</p>
可以看到，模型主要分为如下几个部分：

1. **输入层**：输入样本由原始的英文单词组成，将每个英文单词转换为字典中的id表示。

2. **词向量层**：使用 trainable 的 embedding 矩阵，将原先的 id 表示转换为向量表示。这种将英文单词转换为词向量的方法，比传统的 one-hot 表示更能体现词语的语义内容，关于词向量的更多信息请参考 PaddleBook 中的[词向量](https://github.com/PaddlePaddle/book/tree/develop/04.word2vec)一节。

3. **词向量拼接层**：将词向量进行串联，将向量首尾相接形成一个长向量。

4. **全连接隐层**：将上一层获得的长向量输入一层隐层的神经网络，输出特征向量。

5. **NCE层**：推断时，输出层的神经元数量和样本的类别数一致，在这里就是整个字典的大小，最后使用 softmax 对每个类别的概率做归一化操作，因此第$i$个神经元的输出就可以认为是样本属于第$i$类的预测概率。训练时，我们需要构造二分类分类器。

## 训练阶段
训练直接运行``` python train.py ```。程序第一次运行会检测用户缓存文件夹中是否包含 ptb 数据集，如果未包含，则自动下载。运行过程中，每1000个 iteration 会打印模型训练信息，主要包含训练损失，每个 pass 计算一次测试损失，并同时会保存一次最新的模型。

在 PaddlePaddle 中也有已经实现好的 NCE layer，有一些参数需要自行根据实际场景进行设计：
代码实现如下：

```python
cost = paddle.layer.nce(
    input=hidden_layer,
    label=next_word,
    num_classes=dict_size,
    param_attr=paddle.attr.Param(name='nce_w'),
    bias_attr=paddle.attr.Param(name='nce_b'),
    act=paddle.activation.Sigmoid(),
    num_neg_samples=25,
    neg_distribution=None)
```

| 参数名  | 参数作用  | 介绍 |
|:-------------: |:---------------:| :-------------:|
| param\_attr / bias\_attr | 用来设置参数名字 |         可以方便后面预测阶段好来实现网络的参数共享，具体内容下一个章节里会陈述。|
| num\_neg\_samples | 参数负责控制对负样例的采样个数。        |           可以控制正负样本比例 |
| neg\_distribution | 可以控制生成负样例标签的分布，默认是一个均匀分布。 | 可以自行控制负样本采样时各个类别的权重 |
| act | 表示使用何种激活函数。 | 根据 NCE 的原理，这里应该使用 sigmoid 函数。 |


## 预测阶段
预测直接运行``` python infer.py ```，程序会首先加载最新模型，然后按照 batch 大小依次进行预测，并打印预测结果。预测阶段需要共享 NCE layer 中的逻辑回归训练得到的参数，因为训练和预测计算逻辑不一样，所以需要重新写推断层，推断层的参数为训练时的参数，所以需要参数共享。

具体实现推断层的方法，先是通过 paddle.attr.Param 方法获取参数值，PaddlePaddle 会自行在模型中寻找相同参数名的参数并获取。然后使用 paddle.layer.trans\_full\_matrix\_projection 对隐层输出向量 hidden\_layer 做一个矩阵右乘，从而得到最后的类别向量，将类别向量输入 softmax 做一个归一操作，从而得到最后的类别概率分布。

代码实现如下：

```python
with paddle.layer.mixed(
        size=dict_size,
        act=paddle.activation.Softmax(),
        bias_attr=paddle.attr.Param(name='nce_b')) as prediction:
    prediction += paddle.layer.trans_full_matrix_projection(
        input=hidden_layer, param_attr=paddle.attr.Param(name='nce_w'))
```

## 参考文献
1. Mathematiques C D R. [Quick Training of Probabilistic Neural Nets by Importance Sampling](http://www.iro.umontreal.ca/~lisa/pointeurs/submit_aistats2003.pdf)[C]// 2002.

2. Mnih A, Kavukcuoglu K. [Learning word embeddings efficiently with noise-contrastive estimation](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)[C]//Advances in neural information processing systems. 2013: 2265-2273.

3. Morin, F., & Bengio, Y. (2005, January). [Hierarchical Probabilistic Neural Network Language Model](http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf). In Aistats (Vol. 5, pp. 246-252).

4. Mnih A, Teh Y W. [A Fast and Simple Algorithm for Training Neural Probabilistic Language Models](http://xueshu.baidu.com/s?wd=paperuri%3A%280735b97df93976efb333ac8c266a1eb2%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fabs%2F1206.6426&ie=utf-8&sc_us=5770715420073315630)[J]. Computer Science, 2012:1751-1758.
