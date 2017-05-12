# 文本分类
文本分类是机器学习中的一项常见任务，主要目的是根据一条文本的内容，判断该文本所属的类别。在本例子中，我们利用有标注的IMDB语料库训练二分类DNN和CNN模型，完成对语料的简单文本分类。

DNN与CNN模型之间最大的区别在于，CNN模型中存在卷积结构，而DNN大多使用基本的全连接结构。这使得CNN模型可以对语料信息中相邻单词组成的短语进行分析。例如，"The apple is not bad"，其中的"not bad"是决定这个句子情感的关键。对于DNN模型来说，只能感知到句子中有一个"not"和一个"bad"，而CNN模型则可能直接感知到"not bad"这个关键词组。因此，在大多数文本分类任务上，CNN模型的表现要好于DNN。

## 实验数据
本例子的实验在IMDB数据集上进行（[数据集下载](http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz)）。IMDB数据集包含了来自IMDb（互联网电影数据库）网站的5万条电影影评，并被标注为正面/负面两种评价。数据集被划分为train和test两部分，各2.5万条数据，正负样本的比例基本为1:1。样本直接以英文原文的形式表示。

本样例在第一次运行的时候会自动下载IMDB数据集并缓存，用户无需手动下载。

## DNN模型

**DNN的模型结构入下图所示：**

<p align="center">
<img src="images/dnn_net.png" width = "90%" align="center"/><br/>
图1. DNN文本分类模型
</p>

**可以看到，模型主要分为如下几个部分：**

- **词向量层**：IMDB的样本由原始的英文单词组成，为了方便模型的训练，必须将英文单词转化为固定维度的向量。  

- **最大池化层**：最大池化在时间序列上进行，池化过程消除了不同语料样本在单词数量多少上的差异，并提炼出词向量中每一下标位置上的最大值。经过池化后，样本被转化为一条固定维度的向量。例如，假设最大池化前的矩阵为`[[2,3,5],[7,3,6],[1,4,0]]`，该矩阵每一列代表一个词向量，则最大池化的结果为：`[[5],[7],[4]]`。

- **全连接隐层**：经过最大池化后的向量被送入两个连续的隐层，隐层之间为全连接结构。


- **输出层**：输出层的神经元数量和样本的类别数一致，例如在二分类问题中，输出层会有2个神经元。通过Softmax激活函数，我们保证输出层各神经元的输出之和为1，因此第i个神经元的输出就可以认为是样本属于第i类的预测概率。

**通过PaddlePaddle实现该DNN结构的代码如下：**

```python
import paddle.v2 as paddle

def fc_net(input_dim, class_dim=2, emb_dim=256):
    # input layers
    data = paddle.layer.data("word",
                             paddle.data_type.integer_value_sequence(input_dim))
    lbl = paddle.layer.data("label", paddle.data_type.integer_value(class_dim))

    # embedding layer
    emb = paddle.layer.embedding(input=data, size=emb_dim)
    # max pooling
    seq_pool = paddle.layer.pooling(
        input=emb, pooling_type=paddle.pooling.Max())

    # two hidden layers
    hd_layer_size = [128, 32]
    hd_layer_init_std = [1.0/math.sqrt(s)/3.0 for s in hd_layer_size]
    hd1 = paddle.layer.fc(
        input=seq_pool,
        size=hd_layer_size[0],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=hd_layer_init_std[0]))
    hd2 = paddle.layer.fc(
        input=hd1,
        size=hd_layer_size[1],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=hd_layer_init_std[1]))

    # output layer
    output = paddle.layer.fc(
        input=hd2,
        size=class_dim,
        act=paddle.activation.Softmax(),
        param_attr=paddle.attr.Param(initial_std=1.0/math.sqrt(class_dim)/3.0))

    cost = paddle.layer.classification_cost(input=output, label=lbl)

    return cost, output

```
该DNN模型默认对输入的语料进行二分类（`class_dim=2`），embedding的词向量维度默认为256（`emd_dim=256`），两个隐层均使用Tanh激活函数（`act=paddle.activation.Tanh()`）。

需要注意的是，该模型的输入数据为整数序列，而不是原始的英文单词序列。事实上，为了处理方便我们一般会事先将单词根据词频顺序进行id化，即将单词用整数替代。这一步一般在DNN模型之外完成。

## CNN模型

**CNN的模型结构如下图所示：**

<p align="center">
<img src="images/cnn_net.png" width = "90%" align="center"/><br/>
图2. CNN文本分类模型
</p>

**可以看到，模型主要分为如下几个部分:**

- **词向量层**：与DNN中词向量层的作用一样，将英文单词转化为固定维度的向量。如图2中所示，将得到的词向量定义为行向量，再将语料中所有的单词产生的行向量拼接在一起组成矩阵。假设词向量维度为5，语料“The cat sat on the read mat”包含7个单词，那么得到的矩阵维度为7*5。

- **卷积层**： 文本分类中的卷积在时间序列上进行，即卷积核的宽度和词向量层产出的矩阵一致，卷积验证矩阵的高度方向进行。卷积后得到的结果被称为“特征图”（feature map）。假设卷积核的高度为h，矩阵的高度为N，卷积的步长为1，则得到的特征图为一个高度为N+1-h的向量。可以同时使用多个不同高度的卷积核，得到多个特征图。

- **最大池化层**: 对卷积得到的各个特征图分别进行最大池化操作。由于特征图本身已经是向量，因此这里的最大池化实际上就是简单地选出各个向量中的最大元素。各个最大元素又被并置在一起，组成新的向量，显然，该向量的维度等于特征图的数量，也就是卷积核的数量。

- **全连接与输出层**：将最大池化的结果通过全连接层输出，与DNN模型一样，最后输出层的神经元个数与样本的类别数量一致，且输出之和为1。

**通过PaddlePaddle实现该CNN结构的代码如下：**

```python
import paddle.v2 as paddle

def convolution_net(input_dim, class_dim=2, emb_dim=128, hid_dim=128):
    # input layers
    data = paddle.layer.data("word",
                             paddle.data_type.integer_value_sequence(input_dim))
    lbl = paddle.layer.data("label", paddle.data_type.integer_value(2))

    #embedding layer
    emb = paddle.layer.embedding(input=data, size=emb_dim)

    # convolution layers with max pooling
    conv_3 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=3, hidden_size=hid_dim)
    conv_4 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=4, hidden_size=hid_dim)

    # fc and output layer
    output = paddle.layer.fc(
        input=[conv_3, conv_4], size=class_dim, act=paddle.activation.Softmax())

    cost = paddle.layer.classification_cost(input=output, label=lbl)

    return cost, output

```

该CNN网络的输入数据类型和前面介绍过的DNN一致。`paddle.networks.sequence_conv_pool`为Paddle中已经封装好的带有池化的文本序列卷积模块，该模块的`context_len`参数用于指定卷积核在同一时间覆盖的文本长度，也即图2中的卷积核的高度；`hidden_size`用于指定该类型的卷积核的数量。可以看到，上述代码定义的结构中使用了128个大小为3的卷积核和128个大小为4的卷积核，这些卷积的结果经过最大池化和结果并置后产生一个256维的向量，向量经过一个全连接层输出最终预测结果。

## 运行与输出

本部分以上文介绍的DNN网络为例，介绍如何利用样例中的`text_classification_dnn.py`脚本进行DNN网络的训练和对新样本的预测。

`text_classification_dnn.py`中的代码分为四部分：

- **fc_net函数**：定义dnn网络结构，上文已经有说明。

- **train\_dnn\_model函数**：模型训练函数。定义优化方式、训练输出等内容，并组织训练流程。该函数运行完成前会将训练得到的模型数保存至硬盘上的`dnn_params.tar.gz`文件中。本函数接受一个整数类型的参数，表示训练pass的轮数。

- **dnn_infer函数**：载入已有模型并对新样本进行预测。函数开始运行后会从当前路径下寻找并读取`dnn_params.tar.gz`文件，加载其中的模型参数，并对test数据集中的前100条样本进行预测。

- **main函数**：主函数

要运行本样例，直接在`text_classification_dnn.py`所在路径下执行`python ./text_classification_dnn.py`即可，样例会自动依次执行数据集下载、数据读取、模型训练和保存、模型读取、新样本预测等步骤。

预测的输出形式为：

```
[ 0.99892634  0.00107362] 0
[ 0.00107638  0.9989236 ] 1
[ 0.98185927  0.01814074] 0
[ 0.31667888  0.68332112] 1
[ 0.98853314  0.01146684] 0
```

每一行表示一条样本的预测结果。前两列表示该样本属于0、1这两个类别的预测概率，最后一列表示样本的实际label。
