# Neural Machine Translation with External Memory

带**外部存储或记忆**（External Memory）模块的神经机器翻译模型（Neural Machine Translation, NTM），是神经机器翻译模型的一个重要扩展。它利用可微分（differentiable）的外部记忆模块（其读写控制器以神经网络方式实现），来拓展神经翻译模型内部的工作记忆（Working Memory）的带宽或容量，即作为一个高效的 “外部知识库”，辅助完成翻译等任务中大量信息的临时存储和提取，有效提升模型效果。

该模型不仅可应用于翻译任务，同时可广泛应用于其他需要 “大容量动态记忆” 的自然语言处理和生成任务。例如：机器阅读理解 / 问答（Machine Reading Comprehension / Question Answering)、多轮对话（Multi-turn Dialog）、其他长文本生成任务等。同时，“记忆” 作为认知的重要部分之一，可被用于强化其他多种机器学习模型的表现。该示例仅基于神经机器翻译模型（单指 Seq2Seq， 序列到序列）结合外部记忆机制，起到抛砖引玉的作用，并解释 PaddlePaddle 在搭建此类模型时的灵活性。

本文所采用的外部记忆机制，主要指**神经图灵机**\[[1](#references)\]。值得一提的是，神经图灵机仅仅是神经网络模拟记忆机制的尝试之一。记忆机制长久以来被广泛研究，近年来在深度神经网络的背景下，涌现出一系列有意思的工作。例如：记忆网络（Memory Networks）、可微分神经计算机（Differentiable Neural Computers, DNC）等。除神经图灵机外，其他均不在本文的讨论范围内。

本文的实现主要参考论文\[[2](#references)\]，但略有不同。并基于 PaddlePaddle V2 APIs。初次使用请参考PaddlePaddle [安装教程](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/getstarted/build_and_install/docker_install_cn.rst)。

## Model Overview

### Introduction

记忆（Memory)，是人类（或动物）认知的重要环节之一。记忆赋予认知在时间上的协调性，使得复杂认知（不同于感知）成为可能。记忆，同样是机器学习模型需要拥有的关键能力之一。

可以说，任何机器学习模型，原生就拥有一定的记忆能力，无论它是参数模型，还是非参模型，无论是传统的 SVM（支持向量即记忆），还是神经网络模型（网络参数即记忆）。然而，这里的 “记忆” 绝大部分是指**静态记忆**，即在模型训练结束后，“记忆” 是固化的，在预测时，模型是静态一致的，不拥有额外的跨时间步的记忆能力。

#### 动态记忆 #1 --- RNNs 中的隐状态向量

当我们需要处理带时序的序列认知问题（如自然语言处理、序列决策优化等），我们需要在不同时间步上维持一个相对稳定的信息通路。带有隐状态向量 $h$（Hidden State 或 Cell State $c$）的 RNN 模型 ，即拥有这样的 “**动态记忆**” 能力。每一个时间步，模型均可从 $h$ 或 $c$ 中获取过去时间步的 “记忆” 信息，并可动态地往上叠加新的信息。这些信息在模型推断时随着不同的样本而不同，是 “动态” 的。

注意，尽管 $c$ 或者叠加 Gate 结构的 $h$ 的存在，从优化和概率论的角度看通常都有着不同的解释（优化角度：为了在梯度计算中改良 Jacobian 矩阵的特征值，以减轻长程梯度衰减问题，降低优化难度；概率论角度：使得序列具有马尔科夫性），但不妨碍我们从直觉的角度将它理解为某种 “线性” 的且较 “直接” 的 “记忆通道”。

#### 动态记忆 #2 --- Seq2Seq 中的注意力机制
然而这样的一个向量化的 $h$ 的信息带宽极为有限。在 Seq2Seq 序列到序列生成模型中，这样的带宽瓶颈更表现在信息从编码器（encoder）转移至解码器（decoder）的过程中的信息丢失，即通常所说的 “依赖一个状态向量来编码整个源句子，有着严重的信息损失”。

于是，注意力机制（Attention Mechanism）被提出，用于克服上述困难。在解码时，解码器不再仅仅依赖来自编码器的唯一的状态向量，而是依赖一个向量组（其中的每个向量记录编码器处理每个token时的状态向量），并通过软性的（soft）、广泛分布的（distributed) 的注意强度（attention strength) 来分配注意力资源，提取（线性加权）信息用于序列的不同位置的符号生成。

这里的 “向量组” 蕴含着更大更精准的信息，它可以被认为是一个无界的外部存储器（Unbounded External Memory）。在源语言的编码（encoding）完成时，该外部存储即被初始化为各 token 的状态向量，而在其后的整个解码过程中，只读不写（这是该机制不同于 NTM的地方之一）。同时，读取的过程仅采用基于内容的寻址（Content-based Addressing），而不使用基于位置的寻址 (Location-based Addressing)。当然，这两点局限不是非要如此，仅仅是传统的注意力机制如此，这有待于进一步的探索。另外，这里的 “无界” 指的是 “记忆向量组” 的向量个数非固定，而是随着源语言的 token 数的变化而变化，数量不受限。

#### 动态记忆 #3 --- 神经图灵机

图灵机（Turing Machines)，或冯诺依曼体系（Von Neumann Architecture），计算机体系结构的雏形。运算器（如代数计算）、控制器（如逻辑分支控制）和存储器三者一体，共同构成了当代计算机的核心运行机制。神经图灵机（Neural Turing Machines）\[[1](#references)\]试图利用神经网络模型模拟可微分（于是可通过梯度下降来学习）的图灵机，以实现更复杂的智能。而一般的机器学习模型，大部分忽略了显式存储。神经图灵机正是要弥补这样的潜在缺陷。

<div align="center">
<img src="image/turing_machine_cartoon.gif"><br/>
图1. 图灵机（漫画）。
</div>

图灵机的存储机制，被形象比喻成一个纸带（tape），在这个纸带上有读写头（write/read heads）负责读出或者写入信息，纸袋的移动和读写头则受控制器 （contoller) 控制（见图1）。神经图灵机则以矩阵$M \in \mathcal{R}^{n \times m}$模拟 “纸带”（$n$为记忆槽/记忆向量的数量，$m$为记忆向量的长度），以前馈神经网络（MLP）或者 循环神经网络（RNN）来模拟控制器，在 “纸带” 上实现基于内容和基于位置的寻址（不赘述，请参考论文\[[1](#references)\]），并最终写入或读出信息，供其他网络使用（见图2）。

<div align="center">
<img src="image/neural_turing_machine_arch.png"><br/>
图2. 神经图灵机结构示意图。
</div>

和上述的注意力机制相比，神经图灵机有着很多相同点和不同点。相同在于：均利用外部存储和其上的相关操作，矩阵（或向量组）形式的存储，可微分的寻址方式。不同在于：神经图灵机有读有写（真正意义上的存储器），并且其寻址不仅限于基于内容的寻址，同时结合基于位置的寻址（使得例如 “长序列复制” 等需要 “连续寻址” 的任务更容易），此外它是有界的（bounded)。

#### 三种记忆混合，强化神经机器翻译模型

尽管在一般的 Seq2Seq 模型中，注意力机制都已经是标配。然而，注意机制的外部存储仅仅是用于存储源语言的信息。在解码器内部，信息通路仍然是依赖于 RNN 的状态单向量 $h$ 或 $c$。于是，利用神经图灵机的外部存储机制，来补充（或替换）解码器内部的单向量信息通路，成为自然而然的想法。

当然，我们也可以仅仅通过扩大 $h$ 或 $c$的维度来扩大信息带宽，然而，这样的扩展是以 $O(n^2)$ 的存储（状态-状态转移矩阵）和计算复杂度为代价。而基于神经图灵机的记忆扩展的代价是 $O(n)$的，因为寻址是以记忆槽（Memory Slot）为单位，而控制器的参数结构仅仅是和 $m$（记忆槽的大小）有关。值得注意的是，尽管矩阵拉长了也是向量，但基于状态单向量的记忆读取和写入机制，本质上是**全局**的，而 NTM 的机制是局部的，即读取和写入本质上只在部分记忆槽（尽管实际上是全局写入，但是寻址强度的分布是很锐利的，即真正大的强度仅分布于部分记忆槽），因而可以认为是**局部**的。局部的特性让记忆的存取更干净。

所以，在该实现中，RNNs 原有的状态向量 $h$ 或 $c$、 Seq2Seq 常见的注意力机制，被保留；同时，类似 NTM （简化版，无基于位置的寻址） 的有界外部记忆网络被以补充 $h$ 的形式加入。整体的模型实现则类似于论文\[[2](#references)\]，但有少量差异。同时参考该模型的另外一份基于V1 APIs [配置](https://github.com/lcy-seso/paddle_confs_v1/blob/master/mt_with_external_memory/gru_attention_with_external_memory.conf)， 同样有少量差异。具体讨论于 [Further Discussion](#discussions) 一章。

注意到，在我们的实现中，注意力机制（或无界外部存储）和神经图灵机（或有界外部存储）被实现成相同的 ExternalMemory 类。只是前者是**只读**的， 后者**可读可写**。这样处理仅仅是为了便于统一我们对 “记忆机制” 的理解和认识，同时也提供更简洁和统一的实现版本。

### Architecture
网络总体结构基于传统的 Seq2Seq 结构，即RNNsearch\[[3](#references)\] 结构基础上叠加简化版 NTM\[[1](#references)\]。

- 编码器（encoder）采用标准**双向GRU结构**（非 stack），不再赘述。
- 解码器（decoder）采用和论文\[[2](#references)\] 基本相同的结构，见图3（修改自论文\[[2](#references)\]) 。

<div align="center">
<img src="image/memory_enhanced_decoder.png" width=450><br/>
图3. 通过外部记忆增强的解码器结构示意图。
</div>

解码器结构图，解释如下：

1. $M_{t-1}^B$ 和 $M_t^B$ 为有界外部存储矩阵，前者为上一时间步存储矩阵的状态，后者为当前时间步的状态。$\textrm{read}^B$ 和 $\textrm{write}$ 为对应的读写头（包含其控制器）。$r_t$ 为对应的读出向量。
2. $M^S$ 为无界外部存储矩阵，$\textrm{read}^S$ 为对应的读头（无 写头），二者配合即实现传统的注意力机制。$c_t$ 为对应的读出向量（即 attention context)。
3. 虚线框内（除$M^S$外），整体可视为有界外部存储模块。可以看到，除去该部分，网络结构和 RNNsearch\[[3](#references)\] 基本一致。

## Implementation

TBD

## Getting Started

### Prepare the Training Data

TBD

### Training a Model

TBD

### Generating Sequences

TBD

## Discussions

TBD

## References

1. Alex Graves, Greg Wayne, Ivo Danihelka, [Neural Turing Machines](https://arxiv.org/abs/1410.5401). arXiv preprint arXiv:1410.5401, 2014.
2. Mingxuan Wang, Zhengdong Lu, Hang Li, Qun Liu，[Memory-enhanced Decoder Neural Machine Translation](https://arxiv.org/abs/1606.02003). arXiv preprint arXiv:1606.02003, 2016.
3. Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). arXiv preprint arXiv:1409.0473, 2014.