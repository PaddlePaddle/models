## Transformer

以下是本例的简要目录结构及说明：

```text
.
├── images               # README 文档中的图片
├── predict.py           # 预测脚本
├── reader.py            # 数据读取接口
├── README.md            # 文档
├── train.py             # 训练脚本
├── transformer.py       # 模型定义文件
└── transformer.yaml     # 配置文件
```

## 模型简介

机器翻译（machine translation, MT）是利用计算机将一种自然语言(源语言)转换为另一种自然语言(目标语言)的过程，输入为源语言句子，输出为相应的目标语言的句子。

本项目是机器翻译领域主流模型 Transformer 的 PaddlePaddle 实现， 包含模型训练，预测以及使用自定义数据等内容。用户可以基于发布的内容搭建自己的翻译模型。


## 快速开始

### 安装说明

1. paddle安装

    本项目依赖于 PaddlePaddle 2.0rc及以上版本或适当的develop版本，请参考 [安装指南](https://www.paddlepaddle.org.cn/install/quick) 进行安装

2. 下载代码

    克隆代码库到本地

3. 环境依赖

    该模型使用PaddlePaddle，关于环境依赖部分，请先参考PaddlePaddle[安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)关于环境依赖部分的内容。
    此外，需要另外涉及：
      * attrdict
      * pyyaml


### 数据准备

公开数据集：WMT 翻译大赛是机器翻译领域最具权威的国际评测大赛，其中英德翻译任务提供了一个中等规模的数据集，这个数据集是较多论文中使用的数据集，也是 Transformer 论文中用到的一个数据集。我们也将[WMT'16 EN-DE 数据集](http://www.statmt.org/wmt16/translation-task.html)作为示例提供。

另外我们也整理提供了一份处理好的 WMT'16 EN-DE 数据以供下载使用，其中包含词典（`vocab_all.bpe.32000`文件）、训练所需的 BPE 数据（`train.tok.clean.bpe.32000.en-de`文件）、预测所需的 BPE 数据（`newstest2016.tok.bpe.32000.en-de`等文件）和相应的评估预测结果所需的 tokenize 数据（`newstest2016.tok.de`等文件）。

### 单机训练

### 单机单卡

以提供的英德翻译数据为例，首先需要设置 `transformer.yaml` 中 `world_size` 的值为 1，即卡数，后可以执行以下命令进行模型训练：

```sh
# setting visible devices for training
CUDA_VISIBLE_DEVICES=0 python train.py
```

可以在 transformer.yaml 文件中设置相应的参数。

### 单机多卡

同样，需要设置 `transformer.yaml` 中 `world_size` 的值为 8，即卡数，后执行如下命令可以实现八卡训练：

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py
```

### 模型推断

以英德翻译数据为例，模型训练完成后可以执行以下命令对指定文件中的文本进行翻译：

```sh
# setting visible devices for prediction
export CUDA_VISIBLE_DEVICES=0
python predict.py
```

 由 `predict_file` 指定的文件中文本的翻译结果会输出到 `output_file` 指定的文件。执行预测时需要设置 `init_from_params` 来给出模型所在目录，同样可以使用 `infer_batch_size` 来指定预测的 batch_size 的大小，更多参数的使用可以在 `transformer.yaml` 文件中查阅注释说明并进行更改设置。

 因模型评估与 `predict_file` 写入的顺序相关，故仅有单卡预测。


### 模型评估

预测结果中每行输出是对应行输入的得分最高的翻译，对于使用 BPE 的数据，预测出的翻译结果也将是 BPE 表示的数据，要还原成原始的数据（这里指 tokenize 后的数据）才能进行正确的评估。评估过程具体如下（BLEU 是翻译任务常用的自动评估方法指标）：

```sh
# 还原 predict.txt 中的预测结果为 tokenize 后的数据
sed -r 's/(@@ )|(@@ ?$)//g' predict.txt > predict.tok.txt
# 若无 BLEU 评估工具，需先进行下载
# git clone https://github.com/moses-smt/mosesdecoder.git
# 以英德翻译 newstest2014 测试数据为例
perl gen_data/mosesdecoder/scripts/generic/multi-bleu.perl gen_data/wmt16_ende_data/newstest2014.tok.de < predict.tok.txt
```
可以看到类似如下的结果：
```
BLEU = 26.36, 57.7/32.1/20.0/13.0 (BP=1.000, ratio=1.013, hyp_len=63903, ref_len=63078)
```

## 进阶使用

### 背景介绍

Transformer 是论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中提出的用以完成机器翻译（machine translation, MT）等序列到序列（sequence to sequence, Seq2Seq）学习任务的一种全新网络结构，其完全使用注意力（Attention）机制来实现序列到序列的建模[1]。

相较于此前 Seq2Seq 模型中广泛使用的循环神经网络（Recurrent Neural Network, RNN），使用（Self）Attention 进行输入序列到输出序列的变换主要具有以下优势：

- 计算复杂度小
  - 特征维度为 d 、长度为 n 的序列，在 RNN 中计算复杂度为 `O(n * d * d)` （n 个时间步，每个时间步计算 d 维的矩阵向量乘法），在 Self-Attention 中计算复杂度为 `O(n * n * d)` （n 个时间步两两计算 d 维的向量点积或其他相关度函数），n 通常要小于 d 。
- 计算并行度高
  - RNN 中当前时间步的计算要依赖前一个时间步的计算结果；Self-Attention 中各时间步的计算只依赖输入不依赖之前时间步输出，各时间步可以完全并行。
- 容易学习长程依赖（long-range dependencies）
  - RNN 中相距为 n 的两个位置间的关联需要 n 步才能建立；Self-Attention 中任何两个位置都直接相连；路径越短信号传播越容易。

Transformer 中引入使用的基于 Self-Attention 的序列建模模块结构，已被广泛应用在 Bert [2]等语义表示模型中，取得了显著效果。


### 模型概览

Transformer 同样使用了 Seq2Seq 模型中典型的编码器-解码器（Encoder-Decoder）的框架结构，整体网络结构如图1所示。

<p align="center">
<img src="images/transformer_network.png" height=400 hspace='10'/> <br />
图 1. Transformer 网络结构图
</p>

可以看到，和以往 Seq2Seq 模型不同，Transformer 的 Encoder 和 Decoder 中不再使用 RNN 的结构。

### 模型特点

Transformer 中的 Encoder 由若干相同的 layer 堆叠组成，每个 layer 主要由多头注意力（Multi-Head Attention）和全连接的前馈（Feed-Forward）网络这两个 sub-layer 构成。
- Multi-Head Attention 在这里用于实现 Self-Attention，相比于简单的 Attention 机制，其将输入进行多路线性变换后分别计算 Attention 的结果，并将所有结果拼接后再次进行线性变换作为输出。参见图2，其中 Attention 使用的是点积（Dot-Product），并在点积后进行了 scale 的处理以避免因点积结果过大进入 softmax 的饱和区域。
- Feed-Forward 网络会对序列中的每个位置进行相同的计算（Position-wise），其采用的是两次线性变换中间加以 ReLU 激活的结构。

此外，每个 sub-layer 后还施以 Residual Connection [3]和 Layer Normalization [4]来促进梯度传播和模型收敛。

<p align="center">
<img src="images/multi_head_attention.png" height=300 hspace='10'/> <br />
图 2. Multi-Head Attention
</p>

Decoder 具有和 Encoder 类似的结构，只是相比于组成 Encoder 的 layer ，在组成 Decoder 的 layer 中还多了一个 Multi-Head Attention 的 sub-layer 来实现对 Encoder 输出的 Attention，这个 Encoder-Decoder Attention 在其他 Seq2Seq 模型中也是存在的。

## FAQ

**Q:** 预测结果中样本数少于输入的样本数是什么原因  
**A:** 若样本中最大长度超过 `transformer.yaml` 中 `max_length` 的默认设置，请注意运行时增大 `--max_length` 的设置，否则超长样本将被过滤。

**Q:** 预测时最大长度超过了训练时的最大长度怎么办  
**A:** 由于训练时 `max_length` 的设置决定了保存模型 position encoding 的大小，若预测时长度超过 `max_length`，请调大该值，会重新生成更大的 position encoding 表。


## 参考文献
1. Vaswani A, Shazeer N, Parmar N, et al. [Attention is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)[C]//Advances in Neural Information Processing Systems. 2017: 6000-6010.
2. Devlin J, Chang M W, Lee K, et al. [Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805)[J]. arXiv preprint arXiv:1810.04805, 2018.
3. He K, Zhang X, Ren S, et al. [Deep residual learning for image recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
4. Ba J L, Kiros J R, Hinton G E. [Layer normalization](https://arxiv.org/pdf/1607.06450.pdf)[J]. arXiv preprint arXiv:1607.06450, 2016.
5. Sennrich R, Haddow B, Birch A. [Neural machine translation of rare words with subword units](https://arxiv.org/pdf/1508.07909)[J]. arXiv preprint arXiv:1508.07909, 2015.
