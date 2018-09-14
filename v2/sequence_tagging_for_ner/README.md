运行本目录下的程序示例需要使用PaddlePaddle v0.10.0 版本。如果您的PaddlePaddle安装版本低于此要求，请按照[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)中的说明更新PaddlePaddle安装版本。

---

# 命名实体识别

以下是本例的简要目录结构及说明：

```text
.
├── data                 # 存储运行本例所依赖的数据
│   ├── download.sh
├── images               # README 文档中的图片
├── index.html
├── infer.py             # 测试脚本
├── network_conf.py      # 模型定义
├── reader.py            # 数据读取接口
├── README.md            # 文档
├── train.py             # 训练脚本
└── utils.py             # 定义同样的函数
```


## 简介

命名实体识别（Named Entity Recognition，NER）又称作“专名识别”，是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等，是自然语言处理研究的一个基础问题。NER任务通常包括实体边界识别、确定实体类别两部分，可以将其作为序列标注问题解决。

序列标注可以分为Sequence Classification、Segment Classification和Temporal Classification三类[[1](#参考文献)]，本例只考虑Segment Classification，即对输入序列中的每个元素在输出序列中给出对应的标签。对于NER任务，由于需要标识边界，一般采用[BIO标注方法](http://www.paddlepaddle.org/docs/develop/book/07.label_semantic_roles/index.cn.html)定义的标签集，如下是一个NER的标注结果示例：

<p align="center">
<img src="images/ner_label_ins.png" width="80%" align="center"/><br/>
图1. BIO标注方法示例
</p>

根据序列标注结果可以直接得到实体边界和实体类别。类似的，分词、词性标注、语块识别、[语义角色标注](http://www.paddlepaddle.org/docs/develop/book/07.label_semantic_roles/index.cn.html)等任务都可通过序列标注来解决。使用神经网络模型解决问题的思路通常是：前层网络学习输入的特征表示，网络的最后一层在特征基础上完成最终的任务；对于序列标注问题，通常：使用基于RNN的网络结构学习特征，将学习到的特征接入CRF完成序列标注。实际上是将传统CRF中的线性模型换成了非线性神经网络。沿用CRF的出发点是：CRF使用句子级别的似然概率，能够更好的解决标记偏置问题[[2](#参考文献)]。本例也将基于此思路建立模型。虽然，这里以NER任务作为示例，但所给出的模型可以应用到其他各种序列标注任务中。

由于序列标注问题的广泛性，产生了[CRF](http://www.paddlepaddle.org/docs/develop/book/07.label_semantic_roles/index.cn.html)等经典的序列模型，这些模型大多只能使用局部信息或需要人工设计特征。随着深度学习研究的发展，循环神经网络（Recurrent Neural Network，RNN等 序列模型能够处理序列元素之间前后关联问题，能够从原始输入文本中学习特征表示，而更加适合序列标注任务，更多相关知识可参考PaddleBook中[语义角色标注](https://github.com/PaddlePaddle/book/blob/develop/07.label_semantic_roles/README.cn.md)一课。

## 模型详解

NER任务的输入是"一句话"，目标是识别句子中的实体边界及类别，我们参照论文\[[2](#参考文献)\]仅对原始句子进行了一些简单的预处理工作：将每个词转换为小写，并将原词是否大写另作为一个特征，共同作为模型的输入。模型如图2所示，工作流程如下：

1. 构造输入
 - 输入1是句子序列，采用one-hot方式表示
 - 输入2是大写标记序列，标记了句子中每一个词是否是大写，采用one-hot方式表示；
2. one-hot方式的句子序列和大写标记序列通过词表，转换为实向量表示的词向量序列；
3. 将步骤2中的2个词向量序列作为双向RNN的输入，学习输入序列的特征表示，得到新的特性表示序列；
4. CRF以步骤3中模型学习到的特征为输入，以标记序列为监督信号，实现序列标注。

<p align="center">
<img src="images/ner_network.png" width="40%" align="center"/><br/>
图2. NER 模型网络结构图
</p>


## 数据说明

在本例中，我们以 [CoNLL 2003 NER任务](http://www.clips.uantwerpen.be/conll2003/ner/)为例，原始Reuters数据由于版权原因需另外申请免费下载，请大家按照原网站说明获取。

+ 我们仅在`data`目录下的`train`和`test`文件中放置少数样本用以示例输入数据格式。
+ 本例依赖数据还包括
    1. 输入文本的词典
    2. 为词典中的词语提供预训练好的词向量
    2. 标记标签的词典
   标记标签词典已附在`data`目录中，对应于`data/target.txt`文件。输入文本的词典以及词典中词语的预训练的词向量来自：[Stanford CS224d](http://cs224d.stanford.edu/)课程作业。**为运行本例，请首先在`data`目录下运行`download.sh`脚本下载输入文本的词典和预训练的词向量。** 完成后会将这两个文件一并放入`data`目录下，输入文本的词典和预训练的词向量分别对应：`data/vocab.txt`和`data/wordVectors.txt`这两个文件。

CoNLL 2003原始数据格式如下：

```
U.N.         NNP  I-NP  I-ORG
official     NN   I-NP  O
Ekeus        NNP  I-NP  I-PER
heads        VBZ  I-VP  O
for          IN   I-PP  O
Baghdad      NNP  I-NP  I-LOC
.            .    O     O
```

- 第一列为原始句子序列
- 第二、三列分别为词性标签和句法分析中的语块标签，本例不使用
- 第四列为采用了 I-TYPE 方式表示的NER标签
    - I-TYPE 和 BIO 方式的主要区别在于语块开始标记的使用上，I-TYPE只有在出现相邻的同类别实体时对后者使用B标记，其他均使用I标记），句子之间以空行分隔。

我们在`reader.py`脚本中完成对原始数据的处理以及读取，主要包括下面几个步骤:

1. 从原始数据文件中抽取出句子和标签，构造句子序列和标签序列；
2. 将 I-TYPE 表示的标签转换为 BIO 方式表示的标签；
3. 将句子序列中的单词转换为小写，并构造大写标记序列；
4. 依据词典获取词对应的整数索引。


预处理完成后，一条训练样本包含3个部分作为神经网络的输入信息用于训练：（1）句子序列；（2）首字母大写标记序列；（3）标注序列，下表是一条训练样本的示例：

| 句子序列 | 大写标记序列 | 标注序列 |
| -------- | ------------ | -------- |
| u.n.     | 1            | B-ORG    |
| official | 0            | O        |
| ekeus    | 1            | B-PER    |
| heads    | 0            | O        |
| for      | 0            | O        |
| baghdad  | 1            | B-LOC    |
| .        | 0            | O        |

## 运行
### 编写数据读取接口

自定义数据读取接口只需编写一个 Python 生成器实现从原始输入文本中解析一条训练样本的逻辑。[reader.py](./reader.py) 中的`data_reader`函数实现了读取原始数据返回类型为： `paddle.data_type.integer_value_sequence`的 3 个输入（分别对应：词语在字典的序号、是否为大写、标注结果在字典中的序号）给`network_conf.ner_net`中定义的 3 个 `data_layer` 的功能。

### 训练

1. 运行 `sh data/download.sh`
2. 修改 `train.py` 的 `main` 函数，指定数据路径

    ```python
    main(
          train_data_file="data/train",
          test_data_file="data/test",
          vocab_file="data/vocab.txt",
          target_file="data/target.txt",
          emb_file="data/wordVectors.txt",
          model_save_dir="models/")
    ```

3. 运行命令 `python train.py` ，**需要注意：直接运行使用的是示例数据，请替换真实的标记数据。**

    ```text
    commandline:  --use_gpu=False --trainer_count=1
    Initing parameters..
    Init parameters done.
    Pass 0, Batch 0, Cost 41.430110, {'ner_chunk.precision': 0.01587301678955555, 'ner_chunk.F1-score': 0.028368793427944183, 'ner_chunk.recall': 0.13333334028720856, 'error': 0.939393937587738}
    Test with Pass 0, Batch 0, {'ner_chunk.precision': 0.0, 'ner_chunk.F1-score': 0.0, 'ner_chunk.recall': 0.0, 'error': 0.16260161995887756}
    ```

### 预测
1. 修改 [infer.py](./infer.py) 的 `main` 函数，指定：需要测试的模型的路径、测试数据、字典文件，预测标记文件的路径，默认参数如下：

    ```python
    infer(
        model_path="models/params_pass_0.tar.gz",
        batch_size=2,
        test_data_file="data/test",
        vocab_file="data/vocab.txt",
        target_file="data/target.txt")
    ```

2. 在终端运行 `python infer.py`，开始测试，会看到如下预测结果（以下为训练500个pass所得模型的部分预测结果）：

    ```text
    cricket             O
    -                   O
    leicestershire      B-ORG
    take                O
    over                O
    at                  O
    top                 O
    after               O
    innings             O
    victory             O
    .                   O
    london              B-LOC
    1996-08-30          O
    west                B-MISC
    indian              I-MISC
    all-rounder         O
    phil                B-PER
    simmons             I-PER
    took                O
    four                O

    ```
    输出分为两列，以“\t” 分隔，第一列是输入的词语，第二列是标记结果。多条输入序列之间以空行分隔。


## 参考文献

1. Graves A. [Supervised Sequence Labelling with Recurrent Neural Networks](http://www.cs.toronto.edu/~graves/preprint.pdf)[J]. Studies in Computational Intelligence, 2013, 385.
2. Collobert R, Weston J, Bottou L, et al. [Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)[J]. Journal of Machine Learning Research, 2011, 12(1):2493-2537.
