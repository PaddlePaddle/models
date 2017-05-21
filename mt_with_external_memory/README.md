# 带外部记忆机制的神经机器翻译

带**外部记忆**（External Memory）机制的神经机器翻译模型（Neural Machine Translation, NMT），是神经机器翻译模型的一个重要扩展。它利用可微分的外部记忆网络，来拓展神经翻译模型内部工作记忆（Working Memory）的容量或带宽，即引入一个高效的 “外部知识库”，辅助完成翻译等任务中信息的临时存取，有效改善模型表现。

该模型不仅可应用于翻译任务，同时可广泛应用于其他需要 “大容量动态记忆” 的自然语言处理和生成任务，例如：机器阅读理解 / 问答、多轮对话、其他长文本生成等。同时，“记忆” 作为认知的重要部分之一，可用于强化其他多种机器学习模型的表现。

本文所采用的外部记忆机制，主要指**神经图灵机** \[[1](#参考文献)\]，将于后文详细描述。值得一提的是，神经图灵机仅仅是神经网络模拟记忆机制的尝试之一。记忆机制长久以来被广泛研究，近年来在深度学习的背景下，涌现出一系列有价值的工作，例如：记忆网络（Memory Networks）、可微分神经计算机（Differentiable Neural Computers, DNC）等。除神经图灵机外，其他均不在本文的讨论范围内。

本文的实现主要参考论文\[[2](#参考文献)\]。本文假设读者已充分阅读并理解PaddlePaddle Book中[机器翻译](https://github.com/PaddlePaddle/book/tree/develop/08.machine_translation)一章。


## 模型概述

### 记忆机制简介

记忆（Memory)，是人类认知的重要环节之一。记忆赋予认知在时间上的协调性，使得复杂认知（不同于感知）成为可能。记忆，同样是机器学习模型需要拥有的关键能力之一。

可以说，任何机器学习模型，原生就拥有一定的记忆能力：无论它是参数模型（模型参数即记忆），还是非参模型（样本即记忆）；无论是传统的 SVM（支持向量即记忆），还是神经网络模型（网络连接权值即记忆）。然而，这里的 “记忆” 绝大部分是指**静态记忆**，即在模型训练结束后，“记忆” 是固化的；在预测时，模型是静态一致的，不拥有额外的跨时间步的信息记忆能力。

#### 动态记忆 1 --- RNNs 中的隐状态向量

当我们需要处理带时序的序列认知问题（如自然语言处理、序列决策优化等），我们需要在不同时间步上维持一个可持久的信息通路。带有隐状态向量 $h$（或 LSTM 中的细胞状态向量 $c$）的循环神经网络（Recurrent Neural Networks, RNNs） ，即拥有这样的 “**动态记忆**” 能力。每一个时间步，模型均可从 $h$ 或 $c$ 中获取过去时间步的 “记忆” 信息，并可叠加新的信息。这些信息在模型推断时随着不同的样本而不同，是 “动态” 的。

我们注意到，LSTM 中的细胞状态向量 $c$ 的引入，或者 GRU 中状态向量 $h$ 的以门（Gate）控制的线性跨层结构（Leaky Unit）的引入，从优化的角度看有着不同的理解：即为了梯度计算中各时间步的一阶偏导矩阵（雅克比矩阵）的谱分布更接近单位阵，以减轻长程梯度衰减问题，降低优化难度。但这不妨碍我们从直觉的角度将它理解为增加 “线性通路” 使得 “记忆通道” 更顺畅，如图1（引自[此文](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)）所示的 LSTM 中的细胞状态向量 $c$ 可视为这样一个用于信息持久化的 “线性记忆通道”。

<div align="center">
<img src="image/lstm_c_state.png" width=700><br/>
图1. LSTM 中的细胞状态向量作为 “记忆通道” 示意图
</div>

#### 动态记忆 2 --- Seq2Seq 中的注意力机制

然而这样的一个向量化的 $h$ 或 $c$ 的信息带宽有限。在序列到序列生成模型中，这样的带宽瓶颈更表现在信息从编码器（Encoder）转移至解码器（Decoder）的过程中：仅仅依赖一个有限长度的状态向量来编码整个变长的源语句，有着一定程度的信息丢失。

于是，注意力机制（Attention Mechanism）\[[3](#参考文献)\] 被提出，用于克服上述困难。在解码时，解码器不再仅仅依赖来自编码器的唯一的句级编码向量，而是依赖一个向量组，向量组中的每个向量为编码器的各字符（Tokens）级编码向量（状态向量），并通过一组可学习的注意强度（Attention Weights) 来动态分配注意力资源，以线性加权方式提权信息用于序列的不同位置的符号生成（可参考 PaddlePaddle Book [机器翻译](https://github.com/PaddlePaddle/book/tree/develop/08.machine_translation)一章）。这种注意强度的分布，可看成基于内容的寻址（参考神经图灵机 \[[1](#参考文献)\] 中的寻址描述），即在源语句的不同位置根据其内容获取不同的读取强度，起到一种和源语言 “软对齐（Soft Alignment）” 的作用。

这里的 “向量组” 蕴含着更多更精准的信息，它可以被认为是一个无界的外部记忆模块（Unbounded External Memory）。“无界” 指的是向量组的向量个数非固定，而是随着源语言的字符数的变化而变化，数量不受限。在源语言的编码完成时，该外部存储即被初始化为各字符的状态向量，而在其后的整个解码过程中，只读不写（这是该机制不同于神经图灵机的地方之一）。同时，读取的过程仅采用基于内容的寻址（Content-based Addressing），而不使用基于位置的寻址（Location-based Addressing）。两种寻址方式不赘述，详见 \[[1](#参考文献)\]。当然，这两点局限不是非要如此，仅仅是传统的注意力机制如此，有待进一步的探索。

#### 动态记忆 3 --- 神经图灵机

图灵机（Turing Machines）或冯诺依曼体系（Von Neumann Architecture），是计算机体系结构的雏形。运算器（如代数计算）、控制器（如逻辑分支控制）和存储器三者一体，共同构成了当代计算机的核心运行机制。神经图灵机（Neural Turing Machines）\[[1](#参考文献)\] 试图利用神经网络模型模拟可微分（于是可通过梯度下降来学习）的图灵机，以实现更复杂的智能。而一般的机器学习模型，大部分忽略了显式存储。神经图灵机正是要弥补这样的潜在缺陷。

<div align="center">
<img src="image/turing_machine_cartoon.gif"><br/>
图2. 图灵机结构漫画
</div>

图灵机的存储机制，常被形象比喻成一个纸带（Tape），在这个纸带上有读头（Read Head）和 写头（Write Head）负责读出或者写入信息，纸袋的移动和读写头则受控制器 （Contoller) 控制（见图2，引自[此处](http://www.worldofcomputing.net/theory/turing-machine.html)）。神经图灵机则以矩阵$M \in \mathcal{R}^{n \times m}$模拟 “纸带”，其中 $n$ 为记忆向量（又成记忆槽）的数量，$m$为记忆向量的长度，以前馈神经网络或循环神经网络来模拟控制器，在 “纸带” 上实现基于内容和基于位置的寻址（寻址方式不赘述，请参考论文\[[1](#参考文献)\]），并最终写入或读出信息，供其他网络使用。神经图灵机结构示意图，见图3，引自\[[1](#参考文献)\]。

<div align="center">
<img src="image/neural_turing_machine_arch.png"><br/>
图3. 神经图灵机结构示意图
</div>

和上述的注意力机制相比，神经图灵机有着诸多相同点和不同点。相同在于：均利用矩阵（或向量组）形式的存储，可微分的寻址方式。不同在于：神经图灵机有读有写（是真正意义上的存储器），并且其寻址不仅限于基于内容的寻址，同时结合基于位置的寻址（使得例如 “长序列复制” 等需要 “连续寻址” 的任务更容易），此外它是有界的（Bounded)；而注意机制仅仅有读操作，无写操作，并且仅基于内容寻址，此外它是无界的（Unbounded）。

#### 三种记忆混合，强化神经机器翻译模型

尽管在一般的序列到序列模型中，注意力机制已经是标配。然而，注意机制的外部存储仅仅是用于存储源语言的字符级信息。在解码器内部，信息通路仍然是依赖于 RNN 的状态单向量 $h$ 或 $c$。于是，利用神经图灵机的外部存储机制，来补充解码器内部的单向量信息通路，成为自然而然的想法。

当然，我们也可以仅仅通过扩大 $h$ 或 $c$的维度来扩大信息带宽，然而，这样的扩展是以 $O(n^2)$ 的存储和计算复杂度为代价（状态-状态转移矩阵）。而基于神经图灵机的记忆扩展代价是 $O(n)$的，因为寻址是以记忆槽（Memory Slot）为单位，而控制器的参数结构仅仅是和 $m$（记忆槽的大小）有关。另外值得注意的是，尽管矩阵拉长了也是向量，但基于状态单向量的记忆读取和写入机制，本质上是**全局**的；而神经图灵机的机制是局部的，即读取和写入本质上只在部分记忆槽（尽管实际上是全局写入，但是寻址强度的分布是很锐利的，即真正大的强度仅分布于部分记忆槽），因而可以认为是**局部**的。局部的特性让记忆的存取更干净，干扰更小。

所以，在该示例的实现中，RNN 原有的状态向量和注意力机制被保留；同时，基于简化版的神经图灵机的有界外部记忆机制被引入以补充解码器单状态向量记忆。整体的模型实现参考论文\[[2](#参考文献)\]，但有少量差异，详见[其他讨论](#其他讨论)一章。


### 模型网络结构

网络总体结构在带注意机制的序列到序列结构（即RNNsearch\[[3](##参考文献)\]） 基础上叠加简化版神经图灵机\[[1](#参考文献)\]外部记忆模块。

- 编码器（Encoder）采用标准**双向 GRU 结构**（非 stack），不赘述。
- 解码器（Decoder）采用和论文\[[2](#参考文献)\] 基本相同的结构，见图4（修改自论文\[[2](#参考文献参考文献)\]) 。

<div align="center">
<img src="image/memory_enhanced_decoder.png" width=450><br/>
图4. 通过外部记忆增强的解码器结构示意图
</div>

解码器结构图，解释如下：

1. $M_{t-1}^B$ 和 $M_t^B$ 为有界外部存储矩阵，前者为上一时间步存储矩阵的状态，后者为当前时间步的状态。$\textrm{read}^B$ 和 $\textrm{write}$ 为对应的读写头（包含其控制器）。$r_t$ 为对应的读出向量。
2. $M^S$ 为无界外部存储矩阵，$\textrm{read}^S$ 为对应的读头，二者配合即实现传统的注意力机制。$c_t$ 为对应的读出向量。
3. $y_{t-1}$ 为解码器上一步的输出字符并做词向量（Word Embedding)，作为当前步的输入，$y_t$ 为解码器当前步的解码符号的概率分布。
4. 虚线框内（除$M^S$外），整体可视为有界外部存储模块。可以看到，除去该部分，网络结构和 RNNsearch\[[3](#参考文献)\] 基本一致（略有不一致之处为：用于 attention 的 decoder state 被改进，即叠加了一隐层并引入了 $y_{t-1}$）。


## 算法实现

算法实现的关键部分在辅助类`ExternalMemory` 和模型函数 `memory_enhanced_seq2seq`。

### `ExternalMemory` 类

`ExternalMemory` 类实现通用的简化版**神经图灵机**。相比完整版神经图灵机，该类仅实现了基于内容的寻址（Content Addressing, Interpolation），不包括基于位置的寻址（ Convolutional Shift, Sharpening)。读者可以自行将其补充成为一个完整的神经图灵机。

类结构如下：

```
class ExternalMemory(object):
    __init__(self, name, mem_slot_size, boot_layer, readonly, enable_projection)
    __content_addressing__(self, key_vector)
    __interpolation__(self, head_name, key_vector, addressing_weight)
    __get_adressing_weight__(self, head_name, key_vector)
    write(self, write_key)
    read(self, read_key)
```

神经图灵机的 “外部存储矩阵” 采用 `Paddle.layer.memory`实现，注意这里的`is_seq`需设成`True`，该序列的长度表示记忆槽的数量，`size` 表示记忆槽（向量）的大小。同时依赖一个外部层作为初始化， 记忆槽的数量取决于该层输出序列的长度。因此，该类不仅可用来实现有界记忆（Bounded Memory)，同时可用来实现无界记忆 (Unbounded Memory，即记忆槽数量可变)。

`ExternalMemory`类的寻址逻辑通过 `__content_addressing__` 和 `__interpolation__` 两个私有函数实现。读和写操作通过 `read` 和 `write` 两个函数实现。并且读和写的寻址独立进行，不同于 \[[2](#参考文献)\] 中的二者共享同一个寻址强度，目的是为了使得该类更通用。

为了简单起见，控制器（Controller）未被专门模块化，而是分散在各个寻址和读写函数中，并且采用简单的前馈网络模拟控制器。读者可尝试剥离控制器逻辑并模块化，同时可尝试循环神经网络做控制器。

`ExternalMemory` 类具有只读模式，同时差值寻址操作可关闭。便于用该类等价实现传统的注意力机制。

注意， `ExternalMemory` 只能和 `paddle.layer.recurrent_group`配合使用，具体在用户自定义的 `step` 函数中使用，它不可以单独存在。

### `memory_enhanced_seq2seq` 及相关函数

涉及三个主要函数：

```
memory_enhanced_seq2seq(...)
bidirectional_gru_encoder(...)
memory_enhanced_decoder(...)
```

`memory_enhanced_seq2seq` 函数定义整个带外部记忆机制的序列到序列模型，是模型定义的主调函数。它首先调用`bidirectional_gru_encoder` 对源语言进行编码，然后通过 `memory_enhanced_decoder` 进行解码。

`bidirectional_gru_encoder` 函数实现双向单层 GRU（Gated Recurrent Unit） 编码器。返回两组结果：一组为字符级编码向量序列（包含前后向），一组为整个源语句的句级编码向量（仅后向）。前者用于解码器的注意力机制中记忆矩阵的初始化，后者用于解码器的状态向量的初始化。

`memory_enhanced_decoder` 函数实现通过外部记忆增强的 GRU 解码器。它利用同一个`ExternalMemory` 类实现两种外部记忆模块：

- 无界外部记忆：即传统的注意力机制。利用`ExternalMemory`，打开只读开关，关闭插值寻址。并利用解码器的第一组输出作为 `ExternalMemory` 中存储矩阵的初始化（`boot_layer`）。因此，该存储的记忆槽数目是动态可变的，取决于编码器的字符数。
- 有界外部记忆：利用`ExternalMemory`，关闭只读开关，打开插值寻址。并利用解码器的第一组输出，取均值池化（pooling）后并扩展为指定序列长度后，叠加随机噪声（训练和推断时保持一致），作为 `ExternalMemory` 中存储矩阵的初始化（`boot_layer`）。因此，该存储的记忆槽数目是固定的。

注意到，在我们的实现中，注意力机制（或无界外部存储）和神经图灵机（或有界外部存储）被实现成相同的 `ExternalMemory` 类。前者是**只读**的， 后者**可读可写**。这样处理仅仅是为了便于统一我们对 “注意机制” 和 “记忆机制” 的理解和认识，同时也提供更简洁和统一的实现版本。注意力机制也可以通过 `paddle.networks.simple_attention` 实现。

此外，在该实现中，将 `ExternalMemory` 的 `write` 操作提前至 `read` 之前，以避开潜在的拓扑连接局限，详见 [Issue](https://github.com/PaddlePaddle/Paddle/issues/2061)。我们可以看到，本质上他们是等价的。

## 快速开始

### 数据自定义

数据是通过无参的 `reader()` 迭代器函数，进入训练过程。因此我们需要为训练数据和测试数据分别构造两个 `reader()` 迭代器。`reader()` 函数使用 `yield` 来实现迭代器功能（即可通过 `for instance in reader()` 方式迭代运行）， 例如

```
def reader():
    for instance in data_list:
        yield instance
```

`yield` 返回的每条样本需为三元组，分别包含编码器输入字符列表（即源语言序列，需 ID 化），解码器输入字符列表（即目标语言序列，需 ID 化，且序列右移一位），解码器输出字符列表（即目标语言序列，需 ID 化）。

用户需自行完成字符的切分 (Tokenize) ，并构建字典完成 ID 化。

PaddlePaddle 的接口 [paddle.paddle.wmt14](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/dataset/wmt14.py)， 默认提供了一个经过预处理的、较小规模的 wmt14 英法翻译数据集的子集。并提供了两个reader creator函数如下：

```
paddle.dataset.wmt14.train(dict_size)
paddle.dataset.wmt14.test(dict_size)
```

这两个函数被调用时即返回相应的`reader()`函数，供`paddle.traner.SGD.train`使用。

当我们需要使用其他数据时，可参考 [paddle.paddle.wmt14](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/dataset/wmt14.py) 构造相应的 data creator，并替换 `paddle.dataset.wmt14.train` 和 `paddle.dataset.wmt14.train` 成相应函数名。

### 训练及预测

命令行输入：

```python mt_with_external_memory.py```

即可运行训练脚本（默认训练一轮），训练模型将被定期保存于本地 `params.tar.gz`。训练完成后，将为少量样本生成翻译结果，详见 `infer` 函数。

## 其他讨论

#### 和论文\[[2](#参考文献)\]实现的差异

差异如下：

1. 基于内容的寻址公式不同: 原文为 $a = v^T(WM^B + Us)$，本示例为 $a = v^T \textrm{tanh}(WM^B + Us)$，以保持和 \[[3](#参考文献)\] 中的注意力机制寻址方式一致。
2. 有界外部存储的初始化方式不同: 原文为 $M^B = \sigma(W\sum_{i=0}^{i=n}h_i)/n + V$, $V_{i,j}~\in \mathcal{N}(0, 0.1)$，本示例为 $M^B = \sigma(\frac{1}{n}W\sum_{i=0}^{i=n}h_i) + V$。
3. 外部记忆机制的读和写的寻址逻辑不同：原文二者共享同一个寻址强度，相当于权值联结（Weight Tying）正则。本示例不施加该正则，读和写采用独立寻址。
4. 同时间步内的外部记忆读写次序不同：原文为先读后写，本示例为先写后读，本质等价。

## 参考文献

1. Alex Graves, Greg Wayne, Ivo Danihelka, [Neural Turing Machines](https://arxiv.org/abs/1410.5401). arXiv preprint arXiv:1410.5401, 2014.
2. Mingxuan Wang, Zhengdong Lu, Hang Li, Qun Liu, [Memory-enhanced Decoder Neural Machine Translation](https://arxiv.org/abs/1606.02003). In Proceedings of EMNLP, 2016, pages 278–286.
3. Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). arXiv preprint arXiv:1409.0473, 2014.
