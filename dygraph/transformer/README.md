## 简介

### 任务说明

机器翻译（machine translation, MT）是利用计算机将一种自然语言(源语言)转换为另一种自然语言(目标语言)的过程，输入为源语言句子，输出为相应的目标语言的句子。本示例是机器翻译主流模型 Transformer 的实现和相关介绍。

### 数据集说明

我们使用公开的 [WMT'16 EN-DE 数据集](http://www.statmt.org/wmt16/translation-task.html)训练


可以将下载好的wmt16数据集放在`~/.cache/paddle/dataset/wmt16/`目录下

### 安装说明

1. paddle安装

   本项目依赖于 Paddlepaddle Fluid 1.4.1，请参考安装指南进行安装。

2. 安装代码
3. 环境依赖

### 执行训练：
利用python解释器执行train.py即可

### 执行效果

    W0422 13:25:53.853921 116144 device_context.cc:261] Please NOTE: device: 0, CUDA Capability: 35, Driver API Version: 9.0, Runtime API Version: 8.0
    W0422 13:25:53.861614 116144 device_context.cc:269] device: 0, cuDNN Version: 7.0.

    pass num : 0, batch_id: 10, dy_graph avg loss: [9.033163]
    pass num : 0, batch_id: 20, dy_graph avg loss: [8.869838]
    pass num : 0, batch_id: 30, dy_graph avg loss: [8.635877]
    pass num : 0, batch_id: 40, dy_graph avg loss: [8.460026]
    pass num : 0, batch_id: 50, dy_graph avg loss: [8.293438]
    pass num : 0, batch_id: 60, dy_graph avg loss: [8.138791]
    pass num : 0, batch_id: 70, dy_graph avg loss: [7.9594088]
    pass num : 0, batch_id: 80, dy_graph avg loss: [7.7303553]
    pass num : 0, batch_id: 90, dy_graph avg loss: [7.6716228]
    pass num : 0, batch_id: 100, dy_graph avg loss: [7.611051]
    pass num : 0, batch_id: 110, dy_graph avg loss: [7.4179897]
    pass num : 0, batch_id: 120, dy_graph avg loss: [7.318419]

## 进阶使用

### 模型原理介绍

Transformer 是论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中提出的用以完成机器翻译（machine translation, MT）等序列到序列（sequence to sequence, Seq2Seq）学习任务的一种全新网络结构。其同样使用了 Seq2Seq 任务中典型的编码器-解码器（Encoder-Decoder）的框架结构，但相较于此前广泛使用的循环神经网络（Recurrent Neural Network, RNN），其完全使用注意力（Attention）机制来实现序列到序列的建模，整体网络结构如图1所示。

<p align="center">
<img src="../../PaddleNLP/neural_machine_translation/transformer/images/transformer_network.png" height=400 hspace='10'/> <br />
图 1. Transformer 网络结构图
</p>

Encoder 由若干相同的 layer 堆叠组成，每个 layer 主要由多头注意力（Multi-Head Attention）和全连接的前馈（Feed-Forward）网络这两个 sub-layer 构成。
- Multi-Head Attention 在这里用于实现 Self-Attention，相比于简单的 Attention 机制，其将输入进行多路线性变换后分别计算 Attention 的结果，并将所有结果拼接后再次进行线性变换作为输出。参见图2，其中 Attention 使用的是点积（Dot-Product），并在点积后进行了 scale 的处理以避免因点积结果过大进入 softmax 的饱和区域。
- Feed-Forward 网络会对序列中的每个位置进行相同的计算（Position-wise），其采用的是两次线性变换中间加以 ReLU 激活的结构。

此外，每个 sub-layer 后还施以 [Residual Connection](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) 和 [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf) 来促进梯度传播和模型收敛。

<p align="center">
<img src="../../PaddleNLP/neural_machine_translation/transformer/images/multi_head_attention.png" height=300 hspace='10'/> <br />
图 2. Multi-Head Attention
</p>

Decoder 具有和 Encoder 类似的结构，只是相比于组成 Encoder 的 layer ，在组成 Decoder 的 layer 中还多了一个 Multi-Head Attention 的 sub-layer 来实现对 Encoder 输出的 Attention，这个 Encoder-Decoder Attention 在其他 Seq2Seq 模型中也是存在的。
