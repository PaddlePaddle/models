运行本目录下的程序示例需要使用PaddlePaddle v0.11.0及其以上 版本。如果您的PaddlePaddle安装版本低于此要求，请按照[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)中的说明更新PaddlePaddle安装版本。

---

# 文本标注

以下是本例目录包含的文件以及对应说明:

```text
.
├── images              # 文档中的图片
│   ├── window_net.png
│   └── sentence_net.png
├── infer.py            # 预测脚本
├── network_conf.py     # 本例中涉及的各种网络结构均定义在此文件中，若进一步修改模型结构，请查看此文件
├── reader.py           # 读取数据接口，若使用自定义格式的数据，请查看此文件
├── README.cn.md        # 中文文档
├── README.md           # 英文文档
├── run.sh              # 训练任务运行脚本，直接运行此脚本，将以默认参数开始训练任务
├── train.py            # 训练脚本
└── utils.py            # 定义通用的函数，例如：打印日志、解析命令行参数、构建字典、加载字典等
```

## 简介
文本词性标注(POS,part-of-speach)、命名实体识别(NER,Named Entity Recognition),语义角色标注(SRL,Semantic Role Labeling)任务是自然语言的一类基本标注任务
根据给定一条文本序列，判断该文本序列中各个部分所具有的属性，是自然语言处理领域的一项重要的基础任务。
该模型默认任务为文本词性标注(POS),使用传统的布朗语料(Brown Corpus),流程如下：

1. 下载语料。
2. 清洗，标记。
3. 模型设计。
4. 模型学习效果评估。

训练好的分类器能够**自动判断**新出现的文本序列中各个部分的词性。以上过程也是我们去完成一个新的文本词性标注任务需要遵循的常规流程。可以看到，深度学习方法的巨大优势体现在：**免除复杂的特征的设计，只需要对原始文本进行基础的清理、标注即可**。

"No Free Lunch (NFL)" 是机器学习任务基本原则之一：没有任何一种模型是天生优于其他模型的。模型的设计和选择建立在了解不同模型特性的基础之上，但同时也是一个多次实验评估的过程

本例是基于Ronan Collobert等人的研究成果[Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398)，他们提出了两种进行文本序列分析的模型结构，分别为Window approach network,Sentence approach network。
论文中基于语料WSJ(Wall Street Journal)的词性标注(POS)任务在两种结构的表现精度都在96%以上，而本例的默认语料为布朗语料(Brown Corpus)，其POS任务表现也都在96%以上。

## 模型详解

`network_conf.py` 中包括以下模型：
1. `window_net`： Window approach network 模型，是一个全局线性模型。使用基本的全连接结构。
2. `sentence_net`：Sentence approach network 模型，是一个基础的序列模型，使用cnn结构，考虑局部区域之内的特征。

两者各自的一些特点简单总结如下，详细内容可参考[Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398)：

### 1. Window approach network 模型

**Window approach network 模型结构入下图所示：**

<p align="center">
<img src="images/window_net.png" width = "90%" align="center"/><br/>
图1. 本例中的 Window approach network 序列标注模型
</p>

在 PaddlePaddle 实现该 Window approach network 结构的代码见 `network_conf.py` 中的 `window_net` 函数，模型主要分为如下几个部分：

- **词向量层**：为了更好地表示不同词之间语义上的关系，首先将词语转化为固定维度的向量。训练完成后，词与词语义上的相似程度可以用它们的词向量之间的距离来表示，语义上越相似，距离越近。关于词向量的更多信息请参考PaddleBook中的[词向量](https://github.com/PaddlePaddle/book/tree/develop/04.word2vec)一节。

- **中间线性连接隐层**：将固定size的词向量层进行线性变换，隐层为全连接结构，并使用激活函数。
- **全连接与输出层**：输出层的神经元数量和样本的类别数一致，例如在二分类问题中，输出层会有2个神经元。通过Softmax激活函数，输出结果是一个归一化的概率分布，和为1，因此第$i$个神经元的输出就可以认为是样本属于第$i$类的预测概率。

该 Window approach network 模型默认对输入的语料进行多分类（`class_dim=N`，词性的数量），embedding（词向量）维度默认为32（`emd_dim=32`），隐层均使用tanh激活函数。需要注意的是，该模型的输入数据为固定窗口大小的整数序列，而不是原始的单词序列。事实上，为了处理方便，我们一般会事先将单词根据词频顺序进行 id 化，即将词语转化成在字典中的序号。

### 2. Sentence approach network  模型

**Sentence approach network 模型结构如下图所示：**

<p align="center">
<img src="images/sentence_net.png" width = "90%" align="center"/><br/>
图2. 本例中的 Sentence approach network 序列标注模型
</p>

通过 PaddlePaddle 实现该 Sentence approach network 结构的代码见 `network_conf.py` 中的 `sentence_net` 函数，模型主要分为如下几个部分:

- **词向量层**：与 Window approach network 中词向量层的作用一样，将词语转化为固定维度的向量，利用向量之间的距离来表示词之间的语义相关程度。如图2所示，将得到的词向量定义为行向量，再将语料中所有的单词产生的行向量拼接在一起组成矩阵。假设词向量维度为5，窗口宽度设定为7，得到句子 “The cat sat on the read mat” 含 7 个词语，那么得到的矩阵维度为 7*5。关于词向量的更多信息请参考 PaddleBook 中的[词向量](https://github.com/PaddlePaddle/book/tree/develop/04.word2vec)一节。

- **卷积层**： 文本分类中的卷积在时间序列上进行，即卷积核的宽度和词向量层产出的矩阵一致，卷积沿着矩阵的高度方向进行。卷积后得到的结果被称为“特征图”（feature map）。假设卷积核的高度为 $h$，矩阵的高度为 $N$，卷积的步长为 1，则得到的特征图为一个高度为 $N+1-h$ 的向量。可以同时使用多个不同高度的卷积核，得到多个特征图。

- **最大池化层**: 对卷积得到的各个特征图分别进行最大池化操作。由于特征图本身已经是向量，因此这里的最大池化实际上就是简单地选出各个向量中的最大元素。各个最大元素又被拼接在一起，组成新的向量，显然，该向量的维度等于特征图的数量，也就是卷积核的数量。举例来说，假设我们使用了四个不同的卷积核，卷积产生的特征图分别为：`[2,3,5]`、`[8,2,1]`、`[5,7,7,6]` 和 `[4,5,1,8]`，由于卷积核的高度不同，因此产生的特征图尺寸也有所差异。分别在这四个特征图上进行最大池化，结果为：`[5]`、`[8]`、`[7]`和`[8]`，最后将池化结果拼接在一起，得到`[5,8,7,8]`。

- **中间线性连接隐层**：将最大池化的结果通过全连接层进行线性变换，并使用激活函数。

- **全连接与输出层**：将最大池化的结果通过全连接层输出，与 DNN 模型一样，最后输出层的神经元个数与样本的类别数量一致，且输出之和为 1。

Sentence approach network 网络的输入数据类型和 Window approach network 一致。PaddlePaddle 中已经封装好的带有池化的文本序列卷积模块：`paddle.fluid.nets.sequence_conv_pool`，可直接调用。该模块的 `filter_size` 参数用于指定卷积核在同一时间覆盖的文本长度，即图 2 中的卷积核的宽度。`num_filters` 用于指定该类型的卷积核的数量。本例代码默认使用了 64 个大小为 3 的卷积核，这些卷积的结果经过最大池化后产生一个 64 维的向量，向量经过一个全连接层输出最终的预测结果。

## 使用 PaddlePaddle 内置数据运行

### 如何训练

在终端中执行 `sh run.sh` 以下命令， 将以经典序列标注数据集：`Brown Corpus` 直接运行本例，会看到如下输出：

```text
pass_id: 0, batch 1000, avg_acc: 0.383523, avg_cost: 2.697843
pass_id: 0, batch 2000, avg_acc: 0.562102, avg_cost: 1.937032
pass_id: 0, batch 3000, avg_acc: 0.658544, avg_cost: 1.514453
pass_id: 0, batch 4000, avg_acc: 0.716570, avg_cost: 1.254184
pass_id: 0, batch 5000, avg_acc: 0.754566, avg_cost: 1.081084
pass_id: 0, batch 6000, avg_acc: 0.781755, avg_cost: 0.955819
pass_id: 0, batch 7000, avg_acc: 0.802003, avg_cost: 0.863187
...
```
日志每隔 1000 个 batch 输出一次，输出信息包括：（1）Pass 序号；（2）Batch 序号；（3）依次输出当前 Batch 上评估指标的评估结果。评估指标在配置网络拓扑结构时指定，在上面的输出中，输出了训练样本集之的 精度以及cost指标。

### 如何预测

训练结束后模型默认存储在当前工作目录下，在终端中执行 `python infer.py` ，预测脚本会加载训练好的模型进行预测。

- 默认加载使用 `Brown Corpus`train数据训练一个 Pass 产出的模型对 `Brown Corpus`test 进行测试

会看到如下输出：

```text
s_POS = from/in Kansas/np-tl City/nn-tl Before/in his/pp$ departure/nn ,/, a/at group/nn of/in his/pp$ friends/nns ,/, the/at Reverend/np <UNK>/np among/in them/ppo ,/, had/hvd given/vbn him/ppo a/at luncheon/nn ,/, and/cc <UNK>/np had/hvd seen/vbn advance/nn sheets/nns of/in
p_POS = from/in Kansas/np City/nn-tl Before/in his/pp$ departure/nn ,/, a/at group/nn of/in his/pp$ friends/nns ,/, the/at Reverend/np <UNK>/np among/in them/ppo ,/, had/hvd given/vbn him/ppo a/at luncheon/nn ,/, and/cc <UNK>/np had/hvd seen/vbn advance/nn sheets/nns of/in

```

输出日志每两行是对一条样本预测的结果，以 `--` 分隔，分别是：（1）输入文本，s_POS(source POS)；（2）输出文本，p_POS(prediction POS)。

## 使用自定义数据训练和预测

### 如何训练

1. 数据组织

    假设有如下格式的训练数据：每一行为一条样本，以空格 分隔，分隔后的每一个item以"/"分隔为两部分，第一部分是单词，第二部分是tag。以下是两条示例数据：

    ```
 The/at old-time/jj bridges/nns over/in the/at Merrimac/np-tl River/nn-tl in/in Massachusetts/np are/ber of/in unusual/jj interest/nn in/in many/ap respects/nns ./.
 For/in their/pp$ length/nn ,/, their/pp$ types/nns of/in construction/nn ,/, their/pp$ picturesque/jj settings/nns ,/, and/cc their/pp$ literary/jj associations/nns ,/, they/ppss should/md be/be known/vbn and/cc remembered/vbn ./.
 ```

2. 编写数据读取接口

    自定义数据读取接口只需编写一个 Python 生成器实现**从原始输入文本中解析一条训练样本**的逻辑。

    - 详见本例目录下的 `reader.py` 脚本，`reader.py` 提供了读取测试数据的全部代码。

    接下来，只需要将数据读取函数 `train_reader` 作为参数传递给 `train.py` 脚本中的 `paddle.batch` 接口即可使用自定义数据接口读取数据，调用方式如下：

    ```python
    train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.train_reader(train_data_dir, word_dict, lbl_dict),
                buf_size=1000),
            batch_size=batch_size)
    ```

3. 修改命令行参数

    - 如果将数据组织成示例数据的同样的格式，只需在 `run.sh` 脚本中修改 `train.py` 启动参数，指定 `nn_type` 参数，可以直接运行本例，无需修改数据读取接口 `reader.py`。
    - 执行 `python train.py --help` 可以获取`train.py` 脚本各项启动参数的详细说明，主要参数如下：
        - `nn_type`：选择要使用的模型，目前支持两种：“window” 或者 “sentence”。
        - `train_data_dir`：指定训练数据所在的文件夹，使用自定义数据训练，必须指定此参数，否则使用网络中`Brown corpus`训练，同时默认`test_data_dir`，`word_dict`，和 `label_dict` 参数。
        - `test_data_dir`：指定测试数据所在的文件夹，若不指定将不进行测试，除非使用默认语料。
        - `word_dict`：字典文件所在的路径，若不指定，将从训练数据根据词频统计，自动建立字典。
        - `label_dict`：类别标签字典，用于将字符串类型的类别标签，映射为整数类型的序号。
        - `batch_size`：指定多少条样本后进行一次神经网络的前向运行及反向更新。
        - `num_passes`：指定训练多少个轮次。

### 如何预测

1. 修改 `infer.py` 中以下变量，指定使用的模型、指定测试数据。

    ```python
    model_dir = "./models/sentence_epoch0"  # 指定infer模型所在的目录
    test_data_dir = "./data/brown/test"      # 指定测试文件所在的目录
    word_dict_path = "./data/brown/default_word.dict"     # 指定字典所在的路径
    label_dict_path = "./data/brown/default_label.dict"    # 指定类别标签字典的路径
    ```
2. 在终端中执行 `python infer.py`。

# 参考文献
[1] Collobert R, Weston J, Karlen M, et al. Natural Language Processing (Almost) from Scratch[J]. Journal of Machine Learning Research, 2011, 12(1):2493-2537.
