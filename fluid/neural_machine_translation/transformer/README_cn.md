运行本目录下的程序示例需要使用 PaddlePaddle 最新的 develop branch 版本。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)中的说明更新 PaddlePaddle 安装版本。

---

## Transformer

以下是本例的简要目录结构及说明：

```text
.
├── images               # README 文档中的图片
├── config.py            # 训练、预测以及模型参数配置
├── infer.py             # 预测脚本
├── model.py             # 模型定义
├── optim.py             # learning rate scheduling 计算程序
├── reader.py            # 数据读取接口
├── README.md            # 文档
├── train.py             # 训练脚本
└── util.py              # wordpiece 数据解码工具
```

### 简介

Transformer 是论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中提出的用以完成机器翻译（machine translation, MT）等序列到序列（sequence to sequence, Seq2Seq）学习任务的一种全新网络结构，其完全使用注意力（Attention）机制来实现序列到序列的建模[1]。

相较于此前 Seq2Seq 模型中广泛使用的循环神经网络（Recurrent Neural Network, RNN），使用（Self）Attention 进行输入序列到输出序列的变换主要具有以下优势：

- 计算复杂度小
  - 特征维度为 d 、长度为 n 的序列，在 RNN 中计算复杂度为 `O(n * d * d)` （n 个时间步，每个时间步计算 d 维的矩阵向量乘法），在 Self-Attention 中计算复杂度为 `O(n * n * d)` （n 个时间步两两计算 d 维的向量点积或其他相关度函数），n 通常要小于 d 。
- 计算并行度高
  - RNN 中当前时间步的计算要依赖前一个时间步的计算结果；Self-Attention 中各时间步的计算只依赖输入不依赖之前时间步输出，各时间步可以完全并行。
- 容易学习长程依赖（long-range dependencies）
  - RNN 中相距为 n 的两个位置间的关联需要 n 步才能建立；Self-Attention 中任何两个位置都直接相连；路径越短信号传播越容易。

这些也在机器翻译任务中得到了印证，Transformer 模型在训练时间大幅减少的同时取得了 WMT'14 英德翻译任务 BLEU 值的新高。此外，Transformer 在应用于成分句法分析（Constituency Parsing）任务时也有着不俗的表现，这也说明其具有较高的通用性，容易迁移到其他应用场景中。这些都表明 Transformer 有着广阔的前景。

### 模型概览

Transformer 同样使用了 Seq2Seq 模型中典型的编码器-解码器（Encoder-Decoder）的框架结构，整体网络结构如图1所示。

<p align="center">
<img src="images/transformer_network.png" height=400 hspace='10'/> <br />
图 1. Transformer 网络结构图
</p>

Encoder 由若干相同的 layer 堆叠组成，每个 layer 主要由多头注意力（Multi-Head Attention）和全连接的前馈（Feed-Forward）网络这两个 sub-layer 构成。
- Multi-Head Attention 在这里用于实现 Self-Attention，相比于简单的 Attention 机制，其将输入进行多路线性变换后分别计算 Attention 的结果，并将所有结果拼接后再次进行线性变换作为输出。参见图2，其中 Attention 使用的是点积（Dot-Product），并在点积后进行了 scale 的处理以避免因点积结果过大进入 softmax 的饱和区域。
- Feed-Forward 网络会对序列中的每个位置进行相同的计算（Position-wise），其采用的是两次线性变换中间加以 ReLU 激活的结构。

此外，每个 sub-layer 后还施以 Residual Connection [2]和 Layer Normalization [3]来促进梯度传播和模型收敛。

<p align="center">
<img src="images/multi_head_attention.png" height=300 hspace='10'/> <br />
图 2. Multi-Head Attention
</p>

Decoder 具有和 Encoder 类似的结构，只是相比于组成 Encoder 的 layer ，在组成 Decoder 的 layer 中还多了一个 Multi-Head Attention 的 sub-layer 来实现对 Encoder 输出的 Attention，这个 Encoder-Decoder Attention 在其他 Seq2Seq 模型中也是存在的。


### 数据准备

WMT 数据集是机器翻译领域公认的主流数据集；WMT 英德和英法数据集也是 Transformer 论文中所用数据集，其中英德数据集使用了 BPE（byte-pair encoding）[4]编码的数据，英法数据集使用了 wordpiece [5]的数据。我们这里也将使用 WMT 英德和英法翻译数据，并和论文保持一致使用 BPE 和 wordpiece 的数据，下面给出了使用的方法。对于其他自定义数据，参照下文遵循或转换为类似的数据格式即可。

#### WMT 英德翻译数据

[WMT'16 EN-DE 数据集](http://www.statmt.org/wmt16/translation-task.html)是一个中等规模的数据集。参照论文，英德数据集我们使用 BPE 编码的数据，这能够更好的解决未登录词（out-of-vocabulary，OOV）的问题[4]。用到的 BPE 数据可以参照[这里](https://github.com/google/seq2seq/blob/master/docs/data.md)进行下载（如果希望在自定义数据中使用 BPE 编码，可以参照[这里](https://github.com/rsennrich/subword-nmt)进行预处理），下载后解压，其中 `train.tok.clean.bpe.32000.en` 和 `train.tok.clean.bpe.32000.de` 为使用 BPE 的训练数据（平行语料，分别对应了英语和德语，经过了 tokenize 和 BPE 的处理），`newstest2016.tok.bpe.32000.en` 和 `newstest2016.tok.bpe.32000.de` 等为测试数据（`newstest2016.tok.en` 和 `newstest2016.tok.de` 等则为对应的未使用 BPE 的测试数据），`vocab.bpe.32000` 为相应的词典文件（源语言和目标语言共享该词典文件）。

由于本示例中的数据读取脚本 `reader.py` 默认使用的样本数据的格式为 `\t` 分隔的的源语言和目标语言句子对（默认句子中的词之间使用空格分隔），因此需要将源语言到目标语言的平行语料库文件合并为一个文件，可以执行以下命令进行合并：
```sh
paste -d '\t' train.tok.clean.bpe.32000.en train.tok.clean.bpe.32000.de > train.tok.clean.bpe.32000.en-de
```
此外，下载的词典文件 `vocab.bpe.32000` 中未包含表示序列开始、序列结束和未登录词的特殊符号，可以使用如下命令在词典中加入 `<s>` 、`<e>` 和 `<unk>` 作为这三个特殊符号（用 BPE 表示数据已有效避免了未登录词的问题，这里加入只是做通用处理）。
```sh
sed -i '1i\<s>\n<e>\n<unk>' vocab.bpe.32000
```

#### WMT 英法翻译数据

[WMT'14 EN-FR 数据集](http://www.statmt.org/wmt14/translation-task.html)是一个较大规模的数据集。参照论文，英法数据我们使用 wordpiece 表示的数据，wordpiece 和 BPE 类似同为采用 sub-word units 来解决 OOV 问题的方法[5]。我们提供了已完成预处理的 wordpiece 数据的下载，可以从[这里](http://transformer-data.bj.bcebos.com/wmt14_enfr.tar)下载，其中 `train.wordpiece.en-fr` 为使用 wordpiece 的训练数据，`newstest2014.wordpiece.en-fr` 为测试数据（`newstest2014.tok.en` 和 `newstest2014.tok.fr` 为对应的未经 wordpiece 处理过的测试数据，使用[脚本](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl)进行了 tokenize 的处理），`vocab.wordpiece.en-fr` 为相应的词典文件（源语言和目标语言共享该词典文件）。

提供的英法翻译数据无需进行额外的处理，可以直接使用；需要注意的是，这些用 wordpiece 表示的数据中句子内的 token 之间使用 `\x01` 而非空格进行分隔（因部分 token 内包含空格），这需要在训练时进行指定。

### 模型训练

`train.py` 是模型训练脚本。以英德翻译数据为例，可以执行以下命令进行模型训练：
```sh
python -u train.py \
  --src_vocab_fpath data/vocab.bpe.32000 \
  --trg_vocab_fpath data/vocab.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --train_file_pattern data/train.tok.clean.bpe.32000.en-de \
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

更多模型训练相关的参数则在 `config.py` 中的 `ModelHyperParams` 和 `TrainTaskConfig` 内定义；`ModelHyperParams` 定义了 embedding 维度等模型超参数，`TrainTaskConfig` 定义了 warmup 步数等训练需要的参数。这些参数默认使用了 Transformer 论文中 base model 的配置，如需调整可以在该脚本中进行修改。另外这些参数同样可在执行训练脚本的命令行中设置，传入的配置会合并并覆盖 `config.py` 中的配置，如可以通过以下命令来训练 Transformer 论文中的 big model （如显存不够可适当减小 batch size 的值）：

```sh
python -u train.py \
  --src_vocab_fpath data/vocab.bpe.32000 \
  --trg_vocab_fpath data/vocab.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --train_file_pattern data/train.tok.clean.bpe.32000.en-de \
  --token_delimiter ' ' \
  --use_token_batch True \
  --batch_size 3200 \
  --sort_type pool \
  --pool_size 200000 \
  n_layer 6 \
  n_head 16 \
  d_model 1024 \
  d_inner_hid 4096 \
  n_head 16 \
  prepostprocess_dropout 0.3
```
有关这些参数更详细信息的请参考 `config.py` 中的注释说明。对于英法翻译数据，执行训练和英德翻译训练类似，修改命令中的词典和数据文件为英法数据相应文件的路径，另外要注意的是由于英法翻译数据 token 间不是使用空格进行分隔，需要修改 `token_delimiter` 参数的设置为 `--token_delimiter '\x01'`。

训练时默认使用所有 GPU，可以通过 `CUDA_VISIBLE_DEVICES` 环境变量来设置使用的 GPU 数目。也可以只使用 CPU 训练(通过参数 `--divice CPU` 设置)，训练速度相对较慢。在训练过程中，每隔一定 iteration 后(通过参数 `save_freq` 设置，默认为10000)保存模型到参数 `model_dir` 指定的目录，每个 epoch 结束后也会保存 checkpiont 到 `ckpt_dir` 指定的目录，每个 iteration 将打印如下的日志到标准输出：
```txt
step_idx: 0, epoch: 0, batch: 0, avg loss: 11.059394, normalized loss: 9.682427, ppl: 63538.027344
step_idx: 1, epoch: 0, batch: 1, avg loss: 11.053112, normalized loss: 9.676146, ppl: 63140.144531
step_idx: 2, epoch: 0, batch: 2, avg loss: 11.054576, normalized loss: 9.677609, ppl: 63232.640625
step_idx: 3, epoch: 0, batch: 3, avg loss: 11.046638, normalized loss: 9.669671, ppl: 62732.664062
step_idx: 4, epoch: 0, batch: 4, avg loss: 11.030095, normalized loss: 9.653129, ppl: 61703.449219
step_idx: 5, epoch: 0, batch: 5, avg loss: 11.047491, normalized loss: 9.670525, ppl: 62786.230469
step_idx: 6, epoch: 0, batch: 6, avg loss: 11.044509, normalized loss: 9.667542, ppl: 62599.273438
step_idx: 7, epoch: 0, batch: 7, avg loss: 11.011090, normalized loss: 9.634124, ppl: 60541.859375
step_idx: 8, epoch: 0, batch: 8, avg loss: 10.985243, normalized loss: 9.608276, ppl: 58997.058594
step_idx: 9, epoch: 0, batch: 9, avg loss: 10.993434, normalized loss: 9.616467, ppl: 59482.292969
```

### 模型预测

`infer.py` 是模型预测脚本。以英德翻译数据为例，模型训练完成后可以执行以下命令对指定文件中的文本进行翻译：
```sh
python -u infer.py \
  --src_vocab_fpath data/vocab.bpe.32000 \
  --trg_vocab_fpath data/vocab.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --test_file_pattern data/newstest2016.tok.bpe.32000.en-de \
  --use_wordpiece False \
  --token_delimiter ' ' \
  --batch_size 32 \
  model_path trained_models/iter_199999.infer.model \
  beam_size 4 \
  max_out_len 255
```
和模型训练时类似，预测时也需要设置数据和 reader 相关的参数，并可以执行 `python infer.py --help` 查看这些参数的说明（部分参数意义和训练时略有不同）；同样可以在预测命令中设置模型超参数，但应与模型训练时的设置一致；此外相比于模型训练，预测时还有一些额外的参数，如需要设置 `model_path` 来给出模型所在目录，可以设置 `beam_size` 和 `max_out_len` 来指定 Beam Search 算法的搜索宽度和最大深度（翻译长度），这些参数也可以在 `config.py` 中的 `InferTaskConfig` 内查阅注释说明并进行更改设置。

执行以上预测命令会打印翻译结果到标准输出，每行输出是对应行输入的得分最高的翻译。对于使用 BPE 的英德数据，预测出的翻译结果也将是 BPE 表示的数据，要还原成原始的数据（这里指 tokenize 后的数据）才能进行正确的评估，可以使用以下命令来恢复 `predict.txt` 内的翻译结果到 `predict.tok.txt` 中（无需再次 tokenize 处理）：
```sh
sed -r 's/(@@ )|(@@ ?$)//g' predict.txt > predict.tok.txt
```

对于英法翻译的 wordpiece 数据，执行预测和英德翻译预测类似，修改命令中的词典和数据文件为英法数据相应文件的路径，另外需要注意修改 `token_delimiter` 参数的设置为 `--token_delimiter '\x01'`；同时要修改 `use_wordpiece` 参数的设置为 `--use_wordpiece True`，这会在预测时将翻译得到的 wordpiece 数据还原为原始数据输出。为了使用 tokenize 的数据进行评估，还需要对翻译结果进行 tokenize 的处理，[Moses](https://github.com/moses-smt/mosesdecoder) 提供了一系列机器翻译相关的脚本。执行 `git clone https://github.com/moses-smt/mosesdecoder.git` 克隆 mosesdecoder 仓库后，可以使用其中的 `tokenizer.perl` 脚本对 `predict.txt` 内的翻译结果进行 tokenize 处理并输出到 `predict.tok.txt` 中，如下：
```sh
perl mosesdecoder/scripts/tokenizer/tokenizer.perl -l fr < predict.txt > predict.tok.txt
```

接下来就可以使用参考翻译对翻译结果进行 BLEU 指标的评估了。计算 BLEU 值的脚本也在 Moses 中包含，以英德翻译 `newstest2016.tok.de` 数据为例，执行如下命令：
```sh
perl mosesdecoder/scripts/generic/multi-bleu.perl data/newstest2016.tok.de < predict.tok.txt
```
可以看到类似如下的结果（为单机两卡训练 200K 个 iteration 后模型的预测结果）。
```
BLEU = 33.08, 64.2/39.2/26.4/18.5 (BP=0.994, ratio=0.994, hyp_len=61971, ref_len=62362)
```
目前在未使用 model average 的情况下，英德翻译 base model 八卡训练 100K 个 iteration 后测试 BLEU 值如下：

| 测试集 | newstest2013 | newstest2014 | newstest2015 | newstest2016 |
|-|-|-|-|-|
| BLEU | 25.27 | 26.05 | 28.75 | 33.27 |

英法翻译 base model 八卡训练 100K 个 iteration 后在 `newstest2014` 上测试 BLEU 值为36.。

### 分布式训练

Transformer 模型支持同步或者异步的分布式训练。分布式的配置主要两个方面:

1 命令行配置

  - `--local`，有两个取值，`True`表示单机训练，而`False`表示使用分布式训练。默认为单机训练模式。

  - `--sync`，有两个取值，但只有当`--local`参数为False才会产生影响，其中`True`表示同步训练模式，`False`表示异步训练模式。默认为同步训练模式。

2 环境变量配置

  在分布式训练模式下，会手动配置训练的trainer数量和pserver数量。在网络拓扑上，每一个trainer都会和每一个pserver相连，pserver作为服务端，而trainer作为客户端。下面分pserver和trainer说明具体的参数配置：

1) pserver配置

- `PADDLE_IS_LOCAL=[0|1]` 是否是分布式训练，`0`标识是分布式，`1`标识是单机

- `TRAINING_ROLE=PSERVER` 标识当前节点是pserver

- `POD_IP=ip` 设置当前pserver使用对外服务的地址

- `PADDLE_PORT=port` 设置当前pserver对外服务监听端口号，和`POD_IP`共同构成对外的唯一标识

- `PADDLE_TRAINERS_NUM=num` 设置pserver连接的trainer的数量

下面是配置的示例, 使用两个pserver, 192.168.2.2上的配置如下:
```
export PADDLE_PSERVERS=192.168.2.2,192.168.2.3
export POD_IP=192.168.2.2
export PADDLE_TRAINERS_NUM=2
export TRAINING_ROLE=PSERVER
export PADDLE_IS_LOCAL=0
export PADDLE_PORT=6177
```
192.168.2.3上的配置如下:
```
export PADDLE_PSERVERS=192.168.2.2,192.168.2.3
export POD_IP=192.168.2.3
export PADDLE_TRAINERS_NUM=2
export TRAINING_ROLE=PSERVER
export PADDLE_IS_LOCAL=0
export PADDLE_PORT=6177
```
2) trainer配置

- `PADDLE_IS_LOCAL=[0|1]` 是否是分布式训练，`0`标识是分布式，`1`标识是单机

- `TRAINING_ROLE=TRAINER` 标识当前节点是trainer

- `PADDLE_PSERVERS=[ip1,ip2,……]` 设置pserver的ip地址,用于告知trainer互联的pserver的ip, 使用`,`分割

- `PADDLE_TRAINER_ID=num` 设置当前节点的编号, 编号的取值范围为0到N-1的整数

- `PADDLE_PORT=port` 设置请求的pserver服务端口号

下面是配置的示例, 使用两个trainer, trainer 1上的配置如下:
```
export TRAINING_ROLE=TRAINER
export PADDLE_PSERVERS=192.168.2.2,192.168.2.3
export PADDLE_TRAINERS_NUM=2
export PADDLE_TRAINER_ID=0
export PADDLE_IS_LOCAL=0
export PADDLE_PORT=6177
```
trainer 2上的配置如下:
```
export TRAINING_ROLE=TRAINER
export PADDLE_PSERVERS=192.168.2.2,192.168.2.3
export PADDLE_TRAINERS_NUM=2
export PADDLE_TRAINER_ID=1
export PADDLE_IS_LOCAL=0
export PADDLE_PORT=6177
```

### 参考文献
1. Vaswani A, Shazeer N, Parmar N, et al. [Attention is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)[C]//Advances in Neural Information Processing Systems. 2017: 6000-6010.
2. He K, Zhang X, Ren S, et al. [Deep residual learning for image recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
3. Ba J L, Kiros J R, Hinton G E. [Layer normalization](https://arxiv.org/pdf/1607.06450.pdf)[J]. arXiv preprint arXiv:1607.06450, 2016.
4. Sennrich R, Haddow B, Birch A. [Neural machine translation of rare words with subword units](https://arxiv.org/pdf/1508.07909)[J]. arXiv preprint arXiv:1508.07909, 2015.
5. Wu Y, Schuster M, Chen Z, et al. [Google's neural machine translation system: Bridging the gap between human and machine translation](https://arxiv.org/pdf/1609.08144.pdf)[J]. arXiv preprint arXiv:1609.08144, 2016.
