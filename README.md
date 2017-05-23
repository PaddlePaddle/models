
# models简介

## 词向量

- **介绍：**

    狭义上讲，词向量是指把所有文本样本中的每个单词映射到一个向量。比如，对于文本“A B A E B F G”，我们最终得到单词A对应的向量为[0.1 0.6 -0.5]，单词B对应的向量为[-0.2 0.9 0.7] 。相对于one-hot表示方式，词向量方式更容易计算单词之间的相似性，表示方式更加紧凑。
    广义上讲，词向量也可以应用于普通离散特征，可以充分利用无监督数据，充分捕获特征间的关系，可以有效解决特征稀疏、标签数据缺失、数据噪声等问题。

- **应用领域：**

    词向量是自然语言处理中常见的一个操作，是搜索引擎、广告系统、推荐系统等互联网服务背后常见的基础技术。

- **模型配置说明：**

    [word_embedding](https://github.com/PaddlePaddle/models/tree/develop/word_embedding) |
## 文本生成

- **介绍：**

    文本生成(Sequence to Sequence)，是一种时序对映射的过程，实现了深度学习模型在序列问题中的应用，其中比较突出的是机器翻译和机器人问答。

- **应用领域：**

    文本生成模型可扩展应用于：机器翻译、智能对话与问答、广告创意语料生成、自动编码（如金融画像编码）等业务领域

- **模型配置说明：**

    [sequence_tagging_for_ner](https://github.com/PaddlePaddle/models/tree/develop/sequence_tagging_for_ner) | [seq2seq](https://github.com/PaddlePaddle/models/tree/develop/seq2seq)

## LTR

- **介绍：**

    LTR(learning to rank)是用于解决排序问题的监督学习算法。LTR可分为以下三种：

    - Pointwis：将排序问题转化为多类分类问题或者回归问题。对于检索问题，只考虑给定查询下，单个文档的绝对相关度。
    - PairWise：排序问题被转化成结果对的 回归 、 分类 或 有序分类 的问题。考虑给定查询下，两个文档之间的相对相关度。
    - ListWise：不再将Ranking问题直接形式化为一个分类或者回归问题，考虑给定查询下的文档集合的整体序列。

    PaddlePaddle提供的模型是ListWise的一种实现LambdaRank。

- **应用领域：**

    LTR最标准的应用场景是搜索排序，包括：图片搜索排序、外卖美食搜索排序、App搜索排序、酒店搜索排序。同时，还可以扩展应用于：关键词推荐排序、各类业务榜单排序、个性化推荐排序等。

- **模型配置说明：**

    [LTR](https://github.com/PaddlePaddle/models/tree/develop/ltr) |

## 文本分类

- **介绍：**

    通过深度神经网络模型对文本样本进行分类，支持二分类和多分类。模型包含word embedding步骤，用户可直接将原始文本数据作为输入。

- **应用领域：**

    文本分类可扩展用于以下业务领域：文章质量评估，色情暴力文章识别，评论情绪识别，广告物料风险控制等。

- **模型配置说明：**

    [text_classification](https://github.com/PaddlePaddle/models/tree/develop/text_classification) |
