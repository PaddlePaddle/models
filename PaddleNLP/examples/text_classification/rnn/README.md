# 使用传统Recurrent Neural Networks完成中文文本分类任务

文本分类是NLP应用最广的任务之一，可以被应用到多个领域中，包括但不仅限于：情感分析、垃圾邮件识别、商品评价分类...

一般通过将文本表示成向量后接入分类器，完成文本分类。

如何用向量表征文本，使得向量携带语义信息，是我们关心的重点。

本项目开源了一系列模型用于进行文本建模，用户可通过参数配置灵活使用。效果上，我们基于开源情感倾向分类数据集ChnSentiCorp对多个模型进行评测。

情感倾向分析（Sentiment Classification）是一类常见的文本分类任务。其针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有利的决策支持。可通过 [AI开放平台-情感倾向分析](http://ai.baidu.com/tech/nlp_apply/sentiment_classify) 线上体验。

## 模型简介



本项目通过调用[Seq2Vec](../../../paddlenlp/seq2vec/)中内置的模型进行序列建模，完成句子的向量表示。包含最简单的词袋模型和一系列经典的RNN类模型。

| 模型                                             | 模型介绍                                                     |
| ------------------------------------------------ | ------------------------------------------------------------ |
| BOW（Bag Of Words）                              | 非序列模型，将句子表示为其所包含词的向量的加和               |
| RNN (Recurrent Neural Network)                   | 序列模型，能够有效地处理序列信息                             |
| GRU（Gated Recurrent Unit）                      | 序列模型，能够较好地解决序列文本中长距离依赖的问题           |
| LSTM（Long Short Term Memory）                   | 序列模型，能够较好地解决序列文本中长距离依赖的问题           |
| Bi-LSTM（Bidirectional Long Short Term Memory）  | 序列模型，采用双向LSTM结构，更好地捕获句子中的语义特征       |
| Bi-GRU（Bidirectional Gated Recurrent Unit）     | 序列模型，采用双向GRU结构，更好地捕获句子中的语义特征        |
| Bi-RNN（Bidirectional Recurrent Neural Network） | 序列模型，采用双向RNN结构，更好地捕获句子中的语义特征        |
| Bi-LSTM Attention                                | 序列模型，在双向LSTM结构之上加入Attention机制，结合上下文更好地表征句子语义特征 |
| TextCNN                                          | 序列模型，使用多种卷积核大小，提取局部区域地特征             |

+ BOW（Bag Of Words）模型，是一个非序列模型，使用基本的全连接结构；
+ RNN (Recurrent Neural Network)，序列模型，能够有效地处理序列信息；
+ GRU（Gated Recurrent Unit），序列模型，能够较好地解决序列文本中长距离依赖的问题；
+ LSTM（Long Short Term Memory），序列模型，能够较好地解决序列文本中长距离依赖的问题；
+ Bi-LSTM（Bidirectional Long Short Term Memory），序列模型，采用双向LSTM结构，更好地捕获句子中的语义特征；
+ Bi-GRU（Bidirectional Gated Recurrent Unit），序列模型，采用双向GRU结构，更好地捕获句子中的语义特征；
+ Bi-RNN（Bidirectional Recurrent Neural Network），序列模型，采用双向RNN结构，更好地捕获句子中的语义特征；
+ Bi-LSTM Attention， 序列模型，在双向LSTM结构之上加入Attention机制，结合上下文更好地表征句子语义特征；
+ TextCNN, 序列模型，使用多种卷积核大小，提取局部区域地特征；

| 模型  | dev acc | test acc |
| ---- | ------- | -------- |
| BoW  |  0.8970 | 0.8908   |
| Bi-LSTM  | 0.9098  | 0.8983  |
| Bi-GRU  | 0.9014  | 0.8785  |
| Bi-RNN  | 0.8649  |  0.8504 |
| Bi-LSTM Attention |  0.8992 |  0.8856 |
| TextCNN  | 0.9102  | 0.9107 |

## 快速开始

### 安装说明

* PaddlePaddle 安装

   本项目依赖于 PaddlePaddle 2.0 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

* PaddleNLP 安装

   ```shell
   pip install paddlenlp
   ```

* 环境依赖

   本项目依赖于jieba分词，请在运行本项目之前，安装jieba，如`pip install -U jieba`

   Python的版本要求 3.6+，其它环境请参考 PaddlePaddle [安装说明](https://www.paddlepaddle.org.cn/install/quick/zh/2.0rc-linux-docker) 部分的内容

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
.
├── config.py # 运行配置文件
├── data.py # 数据读取
├── train.py # 训练模型主程序入口，包括训练、评估
├── predict.py # 模型预测
├── model.py # 模型组网
└── README.md # 文档说明
```

### 数据准备

#### 使用PaddleNLP内置数据集

```python
train_dataset = ppnlp.datasets.ChnSentiCorp('train')
dev_dataset = ppnlp.datasets.ChnSentiCorp('dev')
test_dataset = ppnlp.datasets.ChnSentiCorp('test')
```

#### 自定义数据集

训练数据由两列组成："text_a\tlabel"

### 模型训练

在模型训练之前，需要先下载词汇表文件word_dict.txt，用于构造词-id映射关系。

```shell
wget https://paddlenlp.bj.bcebos.com/data/word_dict.txt
```

我们以中文情感分类公开数据集ChnSentiCorp为示例数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证
```shell
# CPU启动
python train.py --vocab_path='./word_dict.txt' --use_gpu=False --network_name=bilstm --lr=5e-4 --batch_size=64 --epochs=5 --save_dir='./checkpoints'

# GPU启动
# CUDA_VSIBLE_DEVICES指定想要利用的GPU卡号，可以是单卡，也可以多卡
# CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch train.py --vocab_path='./word_dict.txt' --use_gpu=True --network_name=bilstm --lr=5e-4 --batch_size=64 --epochs=5 --save_dir='./checkpoints'
```

以上参数表示：

* vocab_path: 词汇表文件路径。
* use_gpu: 是否使用GPU进行训练， 默认为`False`。
* network_name: 模型网络名称，默认为`bilstm_attn`， 可更换为bilstm, bigru, birnn，bow，lstm，rnn，gru，bilstm_attn，textcnn等。
* lr: 学习率， 默认为5e-4。
* batch_size: 运行一个batch大小，默认为64。
* epochs: 训练轮次，默认为5。
* save_dir: 训练保存模型的文件路径。
* init_from_ckpt: 恢复模型训练的断点路径。


程序运行时将会自动进行训练，评估，测试。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── 0.pdopt
├── 0.pdparams
├── 1.pdopt
├── 1.pdparams
├── ...
└── final.pdparams
```

**NOTE:** 如需恢复模型训练，则init_from_ckpt只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt=checkpoints/0`即可，程序会自动加载模型参数`checkpoints/0.pdparams`，也会自动加载优化器状态`checkpoints/0.pdopt`。

### 模型预测

启动预测：
```shell
# CPU启动
python predict.py --vocab_path='./word_dict.txt' --use_gpu=False --network_name=bilstm --params_path=checkpoints/final.pdparams

# GPU启动
# CUDA_VSIBLE_DEVICES指定想要利用的GPU卡号，可以是单卡，也可以多卡
# CUDA_VISIBLE_DEVICES=0 python predict.py --vocab_path='./word_dict.txt' --use_gpu=True --network_name=bilstm --params_path='./checkpoints/final.pdparams'
```

将待预测数据分词完毕后，如以下示例：

```text
这个 宾馆 比较 陈旧 了 ， 特价 的 房间 也 很一般 。 总体来说 一般
怀着 十分 激动 的 心情 放映 ， 可是 看着 看着 发现 ， 在 放映 完毕 后 ， 出现 一集米 老鼠 的 动画片 ！
作为 老 的 四星酒店 ， 房间 依然 很 整洁 ， 相当 不错 。   机场 接机 服务 很好 ， 可以 在 车上 办理 入住 手续 ， 节省 时间 。
```

处理成模型所需的`Tensor`，如可以直接调用`preprocess_prediction_data`函数既可处理完毕。之后传入`predict`函数即可输出预测结果。

如

```text
Data: 这个 宾馆 比较 陈旧 了 ， 特价 的 房间 也 很一般 。 总体来说 一般 	 Lable: negative
Data: 怀着 十分 激动 的 心情 放映 ， 可是 看着 看着 发现 ， 在 放映 完毕 后 ， 出现 一集米 老鼠 的 动画片 ！ 	 Lable: negative
Data: 作为 老 的 四星酒店 ， 房间 依然 很 整洁 ， 相当 不错 。   机场 接机 服务 很好 ， 可以 在 车上 办理 入住 手续 ， 节省 时间 。 	 Lable: positive
```

## 其他

1、如何进行多分类？
本项目采用二分类数据集，如需进行多分类任务，修改类别数目及类别标签列表即可。