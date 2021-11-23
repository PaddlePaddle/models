
# 词向量 N-gram / 5-gram

## 模型概览

### N-gram neural model

在计算语言学中，n-gram是一种重要的文本表示方法，表示一个文本中连续的n个项。基于具体的应用场景，每一项可以是一个字母、单词或者音节。 n-gram模型也是统计语言模型中的一种重要方法，用n-gram训练语言模型时，一般用每个n-gram的历史n-1个词语组成的内容来预测第n个词。

Yoshua Bengio等科学家就于2003年在著名论文 Neural Probabilistic Language Models  中介绍如何学习一个神经元网络表示的词向量模型。文中的神经概率语言模型（Neural Network Language Model，NNLM）通过一个线性映射和一个非线性隐层连接，同时学习了语言模型和词向量，即通过学习大量语料得到词语的向量表达，通过这些向量得到整个句子的概率。用这种方法学习语言模型可以克服维度灾难（curse of dimensionality）,即训练和测试数据不同导致的模型不准。注意：由于“神经概率语言模型”说法较为泛泛，我们在这里不用其NNLM的本名，考虑到其具体做法，本文中称该模型为N-gram neural model。

我们在上文中已经讲到用条件概率建模语言模型，即一句话中第$t$个词的概率和该句话的前$t-1$个词相关。可实际上越远的词语其实对该词的影响越小，那么如果考虑一个n-gram, 每个词都只受其前面`n-1`个词的影响，则有：

$$P(w_1, ..., w_T) = \prod_{t=n}^TP(w_t|w_{t-1}, w_{t-2}, ..., w_{t-n+1})$$

给定一些真实语料，这些语料中都是有意义的句子，N-gram模型的优化目标则是最大化目标函数:

$$\frac{1}{T}\sum_t f(w_t, w_{t-1}, ..., w_{t-n+1};\theta) + R(\theta)$$

其中$f(w_t, w_{t-1}, ..., w_{t-n+1})$表示根据历史n-1个词得到当前词$w_t$的条件概率，$R(\theta)$表示参数正则项。

<p align="center">
       <img src="https://github.com/BAYMAX1314/models/blob/develop/word2vec/images/nnlm.png" width=500><br/>
       图1. N-gram神经网络模型
</p>

图1展示了N-gram神经网络模型，从下往上看，该模型分为以下几个部分：
 - 对于每个样本，模型输入$w_{t-n+1},...w_{t-1}$, 输出句子第t个词为字典中`|V|`个词的概率。

   每个输入词$w_{t-n+1},...w_{t-1}$首先通过映射矩阵映射到词向量$C(w_{t-n+1}),...C(w_{t-1})$。

 - 然后所有词语的词向量连接成一个大向量，并经过一个非线性映射得到历史词语的隐层表示：

    $$g=Utanh(\theta^Tx + b_1) + Wx + b_2$$

    其中，$x$为所有词语的词向量连接成的大向量，表示文本历史特征；$\theta$、$U$、$b_1$、$b_2$和$W$分别为词向量层到隐层连接的参数。$g$表示未经归一化的所有输出单词概率，$g_i$表示未经归一化的字典中第$i$个单词的输出概率。

 - 根据softmax的定义，通过归一化$g_i$, 生成目标词$w_t$的概率为：

  $$P(w_t | w_1, ..., w_{t-n+1}) = \frac{e^{g_{w_t}}}{\sum_i^{|V|} e^{g_i}}$$

 - 整个网络的损失值(cost)为多类分类交叉熵，用公式表示为

   $$J(\theta) = -\sum_{i=1}^N\sum_{c=1}^{|V|}y_k^{i}log(softmax(g_k^i))$$

   其中$y_k^i$表示第$i$个样本第$k$类的真实标签(0或1)，$softmax(g_k^i)$表示第i个样本第k类softmax输出的概率。





## 数据准备

### 数据介绍

本教程使用Penn Treebank （PTB）（经Tomas Mikolov预处理过的版本）数据集。PTB数据集较小，训练速度快，应用于Mikolov的公开语言模型训练工具中。其统计情况如下：

<p align="center">
<table>
    <tr>
        <td>训练数据</td>
        <td>验证数据</td>
        <td>测试数据</td>
    </tr>
    <tr>
        <td>ptb.train.txt</td>
        <td>ptb.valid.txt</td>
        <td>ptb.test.txt</td>
    </tr>
    <tr>
        <td>42068句</td>
        <td>3370句</td>
        <td>3761句</td>
    </tr>
</table>
</p>


### 数据预处理

本章训练的是5-gram模型，表示在PaddlePaddle训练时，每条数据的前4个词用来预测第5个词。PaddlePaddle提供了对应PTB数据集的python包`paddle.dataset.imikolov`，自动做数据的下载与预处理，方便大家使用。

预处理会把数据集中的每一句话前后加上开始符号`<s>`以及结束符号`<e>`。然后依据窗口大小（本教程中为5），从头到尾每次向右滑动窗口并生成一条数据。

如"I have a dream that one day" 一句提供了5条数据：

```text
<s> I have a dream
I have a dream that
have a dream that one
a dream that one day
dream that one day <e>
```

最后，每个输入会按其单词次在字典里的位置，转化成整数的索引序列，作为PaddlePaddle的输入。

## 编程实现

本配置的模型结构如下图所示：

<p align="center">
    <img src="https://github.com/BAYMAX1314/models/blob/develop/word2vec/images/ngram.png" width=400><br/>
    图2. 模型配置中的N-gram神经网络模型
</p>

首先，加载所需要的包：

```python
import paddle
import paddle.fluid as fluid
import numpy
from functools import partial
import math
import os
import sys
from __future__ import print_function
```

然后，定义参数：
```python
EMBED_SIZE = 32  # word vector dimension
HIDDEN_SIZE = 256  # hidden layer dimension
N = 5  # train 5-gram
BATCH_SIZE = 32  # batch size

# can use CPU or GPU
use_cuda = os.getenv('WITH_GPU', '0') != '0'

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)
```

不同于之前的PaddlePaddle v2版本，在新的Fluid版本里，我们不必再手动计算词向量。PaddlePaddle提供了一个内置的方法`fluid.layers.embedding`，我们就可以直接用它来构造 N-gram 神经网络。

- 我们来定义我们的 N-gram 神经网络结构。这个结构在训练和预测中都会使用到。因为词向量比较稀疏，我们传入参数 `is_sparse == True`, 可以加速稀疏矩阵的更新。

```python
def inference_program(is_sparse):
    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    fourth_word = fluid.layers.data(name='fourthw', shape=[1], dtype='int64')

    embed_first = fluid.layers.embedding(
        input=first_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_second = fluid.layers.embedding(
        input=second_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_third = fluid.layers.embedding(
        input=third_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_fourth = fluid.layers.embedding(
        input=fourth_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')

    concat_embed = fluid.layers.concat(
        input=[embed_first, embed_second, embed_third, embed_fourth], axis=1)
    hidden1 = fluid.layers.fc(input=concat_embed,
                              size=HIDDEN_SIZE,
                              act='sigmoid')
    predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')
    return predict_word
```

- 基于以上的神经网络结构，我们可以如下定义我们的`训练`方法

```python
def train_program(is_sparse):
    # The declaration of 'next_word' must be after the invoking of inference_program,
    # or the data input order of train program would be [next_word, firstw, secondw,
    # thirdw, fourthw], which is not correct.
    predict_word = inference_program(is_sparse)
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost
```

- 现在我们可以开始训练啦。如今的版本较之以前就简单了许多。我们有现成的训练和测试集：`paddle.dataset.imikolov.train()`和`paddle.dataset.imikolov.test()`。两者都会返回一个读取器。在PaddlePaddle中，读取器是一个Python的函数，每次调用，会读取下一条数据。它是一个Python的generator。

`paddle.batch` 会读入一个读取器，然后输出一个批次化了的读取器。`event_handler`亦可以一并传入`trainer.train`来时不时的输出每个步骤，批次的训练情况。

```python
def optimizer_func():
    # Note here we need to choose more sophisticated optimizers
    # such as AdaGrad with a decay rate. The normal SGD converges
    # very slowly.
    # optimizer=fluid.optimizer.SGD(learning_rate=0.001),
    return fluid.optimizer.AdagradOptimizer(
        learning_rate=3e-3,
        regularization=fluid.regularizer.L2DecayRegularizer(8e-4))


def train(use_cuda, train_program, params_dirname):
    train_reader = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.imikolov.test(word_dict, N), BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            # We output cost every 10 steps.
            if event.step % 10 == 0:
                outs = trainer.test(
                    reader=test_reader,
                    feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])
                avg_cost = outs[0]

                print("Step %d: Average Cost %f" % (event.step, avg_cost))

                # If average cost is lower than 5.8, we consider the model good enough to stop.
                # Note 5.8 is a relatively high value. In order to get a better model, one should
                # aim for avg_cost lower than 3.5. But the training could take longer time.
                if avg_cost < 5.8:
                    trainer.save_params(params_dirname)
                    trainer.stop()

                if math.isnan(avg_cost):
                    sys.exit("got NaN loss, training failed.")

    trainer = fluid.Trainer(
        train_func=train_program,
        optimizer_func=optimizer_func,
        place=place)

    trainer.train(
        reader=train_reader,
        num_epochs=1,
        event_handler=event_handler,
        feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])
```

- `trainer.train`将会开始训练。从`event_handler`返回的监控情况如下：

```text
Step 0: Average Cost 7.337213
Step 10: Average Cost 6.136128
Step 20: Average Cost 5.766995
...
```

## 模型应用
在模型训练后，我们可以用它做一些预测。

### 预测下一个词
我们可以用我们训练过的模型，在得知之前的 N-gram 后，预测下一个词。

```python
def infer(use_cuda, inference_program, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = fluid.Inferencer(
        infer_func=inference_program, param_path=params_dirname, place=place)

    # Setup inputs by creating 4 LoDTensors representing 4 words. Here each word
    # is simply an index to look up for the corresponding word vector and hence
    # the shape of word (base_shape) should be [1]. The length-based level of
    # detail (lod) info of each LoDtensor should be [[1]] meaning there is only
    # one lod_level and there is only one sequence of one word on this level.
    # Note that lod info should be a list of lists.

    data1 = [[211]]  # 'among'
    data2 = [[6]]    # 'a'
    data3 = [[96]]   # 'group'
    data4 = [[4]]    # 'of'
    lod = [[1]]

    first_word  = fluid.create_lod_tensor(data1, lod, place)
    second_word = fluid.create_lod_tensor(data2, lod, place)
    third_word  = fluid.create_lod_tensor(data3, lod, place)
    fourth_word = fluid.create_lod_tensor(data4, lod, place)

    result = inferencer.infer(
        {
            'firstw': first_word,
            'secondw': second_word,
            'thirdw': third_word,
            'fourthw': fourth_word
        },
        return_numpy=False)

    print(numpy.array(result[0]))
    most_possible_word_index = numpy.argmax(result[0])
    print(most_possible_word_index)
    print([
        key for key, value in word_dict.iteritems()
        if value == most_possible_word_index
    ][0])
```

在经历3分钟的短暂训练后，我们得到如下的预测。我们的模型预测 `among a group of` 的下一个词是`a`。这比较符合文法规律。如果我们训练时间更长，比如几个小时，那么我们会得到的下一个预测是 `workers`。

```text
[[0.00106646 0.0007907  0.00072041 ... 0.00049024 0.00041355 0.00084464]]
6
a
```

整个程序的入口很简单：

```python
def main(use_cuda, is_sparse):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    params_dirname = "word2vec.inference.model"

    train(
        use_cuda=use_cuda,
        train_program=partial(train_program, is_sparse),
        params_dirname=params_dirname)

    infer(
        use_cuda=use_cuda,
        inference_program=partial(inference_program, is_sparse),
        params_dirname=params_dirname)


main(use_cuda=use_cuda, is_sparse=True)
```
