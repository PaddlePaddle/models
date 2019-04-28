## 简介

### 任务说明

机器翻译（machine translation, MT）是利用计算机将一种自然语言(源语言)转换为另一种自然语言(目标语言)的过程，输入为源语言句子，输出为相应的目标语言的句子。本示例是机器翻译主流模型 Transformer 的实现和相关介绍。

### 效果说明

我们使用公开的 [WMT'16 EN-DE 数据集](http://www.statmt.org/wmt16/translation-task.html)训练 Base、Big 两种配置的Transformer 模型后，在相应的测试集上进行评测，效果如下所示：

| 测试集 | newstest2014 | newstest2015 | newstest2016 |
|-|-|-|-|
| Base | 26.35 | 29.07 | 33.30 |
| Big | 27.07 | 30.09 | 34.38 |

## 快速开始

### 安装说明

1. paddle安装

   本项目依赖于 PaddlePaddle Fluid 1.3.1 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

2. 安装代码

    克隆数据集代码库到本地
    ```shell
    git clone https://github.com/PaddlePaddle/models.git
    cd models/PaddleNLP/neural_machine_translation/transformer
    ```

3. 环境依赖

   请参考PaddlePaddle[安装说明](http://paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/index_cn.html)部分的内容

### 开始第一次模型调用

1. 数据准备

	运行 `gen_data.sh` 脚本进行 WMT'16 EN-DE 数据集的下载和预处理（时间较长，建议后台运行）。数据处理过程主要包括 Tokenize 和 [BPE 编码（byte-pair encoding）](https://arxiv.org/pdf/1508.07909)。运行成功后，将会生成文件夹 `gen_data`，其目录结构如下：

    ```text
    .
    ├── wmt16_ende_data              # WMT16 英德翻译数据
    ├── wmt16_ende_data_bpe          # BPE 编码的 WMT16 英德翻译数据
    ├── mosesdecoder                 # Moses 机器翻译工具集，包含了 Tokenize、BLEU 评估等脚本
    └── subword-nmt                  # BPE 编码的代码
    ```

    另外我们也整理提供了一份处理好的 WMT'16 EN-DE 数据以供[下载](https://transformer-res.bj.bcebos.com/wmt16_ende_data_bpe_clean.tar.gz)使用（包含训练所需 BPE 数据和词典以及预测和评估所需的 BPE 数据和 tokenize 的数据）

2. 模型下载

	我们提供了基于 WMT'16 EN-DE 数据训练好的模型以供使用：[base model](https://transformer-res.bj.bcebos.com/base_model.tar.gz) 、[big model](https://transformer-res.bj.bcebos.com/big_model.tar.gz)

3. 模型预测

	使用以上提供的数据和模型，可以按照以下代码进行预测，翻译结果将打印到标准输出:
	```sh
    # base model
    python -u infer.py \
    --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
    --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
    --special_token '<s>' '<e>' '<unk>' \
    --test_file_pattern gen_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
    --token_delimiter ' ' \
    --batch_size 32 \
    model_path trained_models/iter_100000.infer.model \
    beam_size 5 \
    max_out_len 255


    # big model
    python -u infer.py \
    --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
    --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
    --special_token '<s>' '<e>' '<unk>' \
    --test_file_pattern gen_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
    --token_delimiter ' ' \
    --batch_size 32 \
    model_path trained_models/iter_100000.infer.model \
    n_head 16 \
    d_model 1024 \
    d_inner_hid 4096 \
    prepostprocess_dropout 0.3 \
    beam_size 5 \
    max_out_len 255
	```
4. 模型评估

	预测结果中每行输出是对应行输入的得分最高的翻译，对于使用 BPE 的数据，预测出的翻译结果也将是 BPE 表示的数据，要还原成原始的数据（这里指 tokenize 后的数据）才能进行正确的评估。评估过程具体如下（BLEU 是翻译任务常用的自动评估方法指标）：

    ```sh
    # 还原 predict.txt 中的预测结果为 tokenize 后的数据
    sed -r 's/(@@ )|(@@ ?$)//g' predict.txt > predict.tok.txt
    # 若无 BLEU 评估工具，需先进行下载
    # git clone https://github.com/moses-smt/mosesdecoder.git
    # 以英德翻译 newstest2014 测试数据为例
    perl gen_data/mosesdecoder/scripts/generic/multi-bleu.perl gen_data/wmt16_ende_data/newstest2014.tok.de < predict.tok.txt
    ```

    完成后可以看到类似如下的结果：
    ```
    BLEU = 26.35, 57.7/32.1/20.0/13.0 (BP=1.000, ratio=1.013, hyp_len=63903, ref_len=63078)
    ```

## 进阶使用

### 模型原理介绍

Transformer 是论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中提出的用以完成机器翻译（machine translation, MT）等序列到序列（sequence to sequence, Seq2Seq）学习任务的一种全新网络结构。其同样使用了 Seq2Seq 任务中典型的编码器-解码器（Encoder-Decoder）的框架结构，但相较于此前广泛使用的循环神经网络（Recurrent Neural Network, RNN），其完全使用注意力（Attention）机制来实现序列到序列的建模，整体网络结构如图1所示。

<p align="center">
<img src="images/transformer_network.png" height=400 hspace='10'/> <br />
图 1. Transformer 网络结构图
</p>

Encoder 由若干相同的 layer 堆叠组成，每个 layer 主要由多头注意力（Multi-Head Attention）和全连接的前馈（Feed-Forward）网络这两个 sub-layer 构成。
- Multi-Head Attention 在这里用于实现 Self-Attention，相比于简单的 Attention 机制，其将输入进行多路线性变换后分别计算 Attention 的结果，并将所有结果拼接后再次进行线性变换作为输出。参见图2，其中 Attention 使用的是点积（Dot-Product），并在点积后进行了 scale 的处理以避免因点积结果过大进入 softmax 的饱和区域。
- Feed-Forward 网络会对序列中的每个位置进行相同的计算（Position-wise），其采用的是两次线性变换中间加以 ReLU 激活的结构。

此外，每个 sub-layer 后还施以 [Residual Connection](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) 和 [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf) 来促进梯度传播和模型收敛。

<p align="center">
<img src="images/multi_head_attention.png" height=300 hspace='10'/> <br />
图 2. Multi-Head Attention
</p>

Decoder 具有和 Encoder 类似的结构，只是相比于组成 Encoder 的 layer ，在组成 Decoder 的 layer 中还多了一个 Multi-Head Attention 的 sub-layer 来实现对 Encoder 输出的 Attention，这个 Encoder-Decoder Attention 在其他 Seq2Seq 模型中也是存在的。

### 代码结构说明

以下是本例的简要目录结构及说明:

```text
.
├── images               # README 文档中的图片
├── config.py            # 训练、预测以及模型参数配置
├── infer.py             # 预测脚本
├── reader.py            # 数据读取接口
├── README.md            # 文档
├── train.py             # 训练脚本
└── gen_data.sh          # 数据生成脚本
```

### 数据格式说明

本示例程序中支持的数据格式为制表符 `\t` 分隔的源语言和目标语言句子对，句子中的 token 之间使用空格分隔
。如需使用 BPE 编码，亦可以使用类似 WMT'16 EN-DE 原始数据的格式，参照 `gen_data.sh` 进行处理。

### 如何训练

数据准备完成后，可以使用 `train,py` 脚本进行训练。以提供的 WMT'16 EN-DE 数据为例，具体如下：

```sh
python -u train.py \
  --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --train_file_pattern gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
  --token_delimiter ' ' \
  --use_token_batch True \
  --batch_size 4096 \
  --sort_type pool \
  --pool_size 200000
```

上述命令中设置了源语言词典文件路径（`src_vocab_fpath`）、目标语言词典文件路径（`trg_vocab_fpath`）、训练数据文件（`train_file_pattern`，支持通配符）等数据相关的参数和构造 batch 方式（`use_token_batch` 指定了数据按照 token 数目或者 sequence 数目组成 batch）等 reader 相关的参数。有关这些参数更详细的信息可以通过执行以下命令查看：

```sh
python train.py --help
```

更多模型训练相关的参数则在 `config.py` 中的 `ModelHyperParams` 和 `TrainTaskConfig` 内定义；`ModelHyperParams` 定义了 embedding 维度等模型超参数，`TrainTaskConfig` 定义了 warmup 步数等训练需要的参数。这些参数默认使用了 Transformer 论文中 base model 的配置，如需调整可以在该脚本中进行修改。另外这些参数同样可在执行训练脚本的命令行中设置，传入的配置会合并并覆盖 `config.py` 中的配置，如可以通过以下命令来训练 Transformer 论文中的 big model ：

```sh
python -u train.py \
  --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --train_file_pattern gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
  --token_delimiter ' ' \
  --use_token_batch True \
  --batch_size 3200 \
  --sort_type pool \
  --pool_size 200000 \
  n_head 16 \
  d_model 1024 \
  d_inner_hid 4096 \
  prepostprocess_dropout 0.3
```

注意，如训练时更改了模型配置，使用 `infer.py` 预测时需要使用对应相同的模型配置；另外，训练时默认使用所有 GPU，可以通过 `CUDA_VISIBLE_DEVICES` 环境变量来设置使用指定的 GPU。

## 其他

### 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
