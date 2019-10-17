## 简介

### 任务说明

机器翻译（machine translation, MT）是利用计算机将一种自然语言(源语言)转换为另一种自然语言(目标语言)的过程，输入为源语言句子，输出为相应的目标语言的句子。本示例是机器翻译主流模型 Transformer 的实现和相关介绍。

动态图文档请见[Dygraph](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/user_guides/howto/dygraph/DyGraph.html)

### 数据集说明

我们使用[WMT-16](http://www.statmt.org/wmt16/)新增的[multimodal task](http://www.statmt.org/wmt16/multimodal-task.html)中的[translation task](http://www.statmt.org/wmt16/multimodal-task.html#task1)的数据集作为示例。该数据集为英德翻译数据，包含29001条训练数据，1000条测试数据。

该数据集内置在了Paddle中，可以通过 `paddle.dataset.wmt16` 使用，执行本项目中的训练代码数据集将自动下载到 `~/.cache/paddle/dataset/wmt16/` 目录下。

### 安装说明

1. paddle安装

   本项目依赖于 PaddlePaddle Fluid 1.6.0 及以上版本（1.6.0 待近期正式发版，可先使用 develop），请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

2. 环境依赖

   多卡运行需要 NCCL 2.4.7 版本。

### 执行训练：
如果是使用GPU单卡训练，启动训练的方式:
```
env CUDA_VISIBLE_DEVICES=0 python train.py
```

这里`CUDA_VISIBLE_DEVICES=0`表示是执行在0号设备卡上，请根据自身情况修改这个参数。如需调整其他模型及训练参数，可在 `config.py` 中修改或使用如下方式传入：

```sh
python train.py \
  n_head 16 \
  d_model 1024 \
  d_inner_hid 4096 \
  prepostprocess_dropout 0.3
```

Paddle动态图支持多进程多卡进行模型训练，启动训练的方式：
```
python -m paddle.distributed.launch --started_port 9999 --selected_gpus=0,1,2,3 --log_dir ./mylog train.py --use_data_parallel 1
```
此时，程序会将每个进程的输出log导入到`./mylog`路径下：
```
.
├── mylog
│   ├── workerlog.0
│   ├── workerlog.1
│   ├── workerlog.2
│   └── workerlog.3
├── README.md
└── train.py
```

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


### 执行预测

训练完成后，使用如下命令进行预测：

```
env CUDA_VISIBLE_DEVICES=0 python predict.py
```

预测结果将输出到 `predict.txt` 文件中（可在运行时通过 `--output_file` 更改），其他模型与预测参数也可在 `config.py` 中修改或使用如下方式传入：

```sh
python predict.py \
  n_head 16 \
  d_model 1024 \
  d_inner_hid 4096 \
  prepostprocess_dropout 0.3
```

完成预测后，可以借助第三方工具进行 BLEU 指标的评估，可按照如下方式进行：

```sh
# 提取 reference 数据
tar -zxvf ~/.cache/paddle/dataset/wmt16/wmt16.tar.gz
awk 'BEGIN {FS="\t"}; {print $2}' wmt16/test > ref.de

# clone mosesdecoder代码
git clone https://github.com/moses-smt/mosesdecoder.git

# 进行评估
perl mosesdecoder/scripts/generic/multi-bleu.perl ref.de < predict.txt
```

使用默认配置单卡训练20个 epoch 训练的模型约有如下评估结果：
```
BLEU = 32.38, 64.3/39.1/25.9/16.9 (BP=1.000, ratio=1.001, hyp_len=12122, ref_len=12104)
```


## 进阶使用

### 自定义数据

- 训练：
  
  修改 `train.py` 中的如下代码段

  ```python
        reader = paddle.batch(wmt16.train(ModelHyperParams.src_vocab_size,
                                          ModelHyperParams.trg_vocab_size),
                              batch_size=TrainTaskConfig.batch_size)
  ```
  
  
  将其中的 `wmt16.train` 替换为类似如下的 python generator ：

  ```python
  def reader(file_name, src_dict, trg_dict):
    start_id = src_dict[START_MARK]  # BOS
    end_id = src_dict[END_MARK]  # EOS
    unk_id = src_dict[UNK_MARK]  # UNK

    src_col, trg_col = 0, 1

    for line in open(file_name, "r"):
        line = line.strip()
        line_split = line.strip().split("\t")
        if len(line_split) != 2:
            continue
        src_words = line_split[src_col].split()
        src_ids = [start_id] + [
            src_dict.get(w, unk_id) for w in src_words
        ] + [end_id]

        trg_words = line_split[trg_col].split()
        trg_ids = [trg_dict.get(w, unk_id) for w in trg_words]

        trg_ids_next = trg_ids + [end_id]
        trg_ids = [start_id] + trg_ids

        yield src_ids, trg_ids, trg_ids_next
  ```

该 generator 产生的数据为单个样本，是包含源句（src_ids），目标句（trg_ids）和标签（trg_ids_next）三个 integer list 的 tuple；其中 src_ids 包含 BOS 和 EOS 的 id，trg_ids 包含 BOS 的 id，trg_ids_next 包含 EOS 的 id。

- 预测：
  修改 `predict.py` 中的如下代码段

  ```python
        reader = paddle.batch(wmt16.test(ModelHyperParams.src_vocab_size,
                                         ModelHyperParams.trg_vocab_size),
                              batch_size=InferTaskConfig.batch_size)
        id2word = wmt16.get_dict("de",
                                 ModelHyperParams.trg_vocab_size,
                                 reverse=True)
  ```

  将其中的 `wmt16.test` 替换为和训练部分类似的 python generator ；另外还需要提供将 id 映射到 word 的 python dict 作为 `id2word` .

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
