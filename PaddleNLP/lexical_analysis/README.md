# 中文词法分析

## 1. 简介

Lexical Analysis of Chinese，简称 LAC，是一个联合的词法分析模型，能整体性地完成中文分词、词性标注、专名识别任务。我们在自建的数据集上对分词、词性标注、专名识别进行整体的评估效果，具体数值见下表；此外，我们在百度开放的 [ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE) 模型上 finetune，并对比基线模型、BERT finetuned 和 ERNIE finetuned 的效果，可以看出会有显著的提升。可通过 [AI开放平台-词法分析](http://ai.baidu.com/tech/nlp/lexical) 线上体验百度的词法分析服务。

|模型|Precision|Recall|F1-score|
|:-:|:-:|:-:|:-:|
|Lexical Analysis|88.0%|88.7%|88.4%|
|BERT finetuned|90.2%|90.4%|90.3%|
|ERNIE finetuned|92.0%|92.0%|92.0%|

## 2. 快速开始

### 安装说明

#### 1.PaddlePaddle 安装

本项目依赖 PaddlePaddle 1.3.2 及以上版本，安装请参考官网 [快速安装](http://www.paddlepaddle.org/paddle#quick-start)。

#### 2. 克隆代码
克隆工具集代码库到本地
```bash
 git clone https://github.com/PaddlePaddle/models.git
 cd models/PaddleNLP/lexical_analysis
```
### 数据准备
下载数据集文件，解压后会生成 `./data/` 文件夹
```bash
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lexical_analysis-dataset-1.0.0.tar.gz
tar xvf lexical_analysis-dataset-1.0.0.tar.gz
```

### 模型下载
我们开源了在自建数据集上训练的词法分析模型，可供用户直接使用，这里提供两种下载方式：

方式一：基于 PaddleHub 命令行工具，PaddleHub 的安装参考 [PaddleHub](https://github.com/PaddlePaddle/PaddleHub)
```bash
# download baseline model
hub download lexical_analysis
tar xvf lexical_analysis-1.0.0.tar.gz

# download ERNIE finetuned model
hub download lexical_analysis_finetuned
tar xvf lexical_analysis_finetuned-1.0.0.tar.gz
```

方式二：直接下载
```bash
# download baseline model
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lexical_analysis-1.0.0.tar.gz
tar xvf lexical_analysis-1.0.0.tar.gz

# download ERNIE finetuned model
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/lexical_analysis_finetuned-1.0.0.tar.gz
tar xvf lexical_analysis_finetuned-1.0.0.tar.gz
```

注：下载 ERNIE 开放的模型请参考 [ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)，下载后可放在 `./pretrained/` 目录下。

### 模型评估
我们基于自建的数据集训练了一个词法分析的模型，可以直接用这个模型对测试集 `./data/test.tsv` 进行验证，
```bash
# baseline model
sh run.sh eval

# ERNIE finetuned model
sh run_ernie.sh eval
```

### 模型训练
基于示例的数据集，可以运行下面的命令，在训练集 `./data/train.tsv` 上进行训练
```bash
# baseline model
sh run.sh train

# ERNIE finetuned model
sh run_ernie.sh train
```

### 模型预测
加载已有的模型，对未知的数据进行预测
```bash
# baseline model
sh run.sh infer

# ERNIE finetuned model
sh run_ernie.sh infer
```

## 3. 进阶使用

### 任务定义与建模
词法分析任务的输入是一个字符串（我们后面使用『句子』来指代它），而输出是句子中的词边界和词性、实体类别。序列标注是词法分析的经典建模方式。我们使用基于 GRU 的网络结构学习特征，将学习到的特征接入 CRF 解码层完成序列标注。CRF 解码层本质上是将传统 CRF 中的线性模型换成了非线性神经网络，基于句子级别的似然概率，因而能够更好的解决标记偏置问题。模型要点如下，具体细节请参考 `run_sequence_labeling.py` 代码。
1. 输入采用 one-hot 方式表示，每个字以一个 id 表示
2. one-hot 序列通过字表，转换为实向量表示的字向量序列；
3. 字向量序列作为双向 GRU 的输入，学习输入序列的特征表示，得到新的特性表示序列，我们堆叠了两层双向GRU以增加学习能力；
4. CRF 以 GRU 学习到的特征为输入，以标记序列为监督信号，实现序列标注。

词性和专名类别标签集合如下表，其中词性标签 24 个（小写字母），专名类别标签 4 个（大写字母）。这里需要说明的是，人名、地名、机名和时间四个类别，在上表中存在两套标签（PER / LOC / ORG / TIME 和 nr / ns / nt / t），被标注为第二套标签的词，是模型判断为低置信度的人名、地名、机构名和时间词。开发者可以基于这两套标签，在四个类别的准确、召回之间做出自己的权衡。

| 标签 | 含义     | 标签 | 含义     | 标签 | 含义     | 标签 | 含义     |
| ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- |
| n    | 普通名词 | f    | 方位名词 | s    | 处所名词 | t    | 时间     |
| nr   | 人名     | ns   | 地名     | nt   | 机构名   | nw   | 作品名   |
| nz   | 其他专名 | v    | 普通动词 | vd   | 动副词   | vn   | 名动词   |
| a    | 形容词   | ad   | 副形词   | an   | 名形词   | d    | 副词     |
| m    | 数量词   | q    | 量词     | r    | 代词     | p    | 介词     |
| c    | 连词     | u    | 助词     | xc   | 其他虚词 | w    | 标点符号 |
| PER  | 人名     | LOC  | 地名     | ORG  | 机构名   | TIME | 时间     |

### 模型原理介绍
上面介绍的模型原理如下图所示：<br />


![GRU-CRF-MODEL](./gru-crf-model.png)

### 数据格式
训练使用的数据可以由用户根据实际的应用场景，自己组织数据。除了第一行是 `text_a\tlabel` 固定的开头，后面的每行数据都是由两列组成，以制表符分隔，第一列是 utf-8 编码的中文文本，以 `\002` 分割，第二列是对应每个字的标注，以 `\002` 分隔。我们采用 IOB2 标注体系，即以 X-B 作为类型为 X 的词的开始，以 X-I 作为类型为 X 的词的持续，以 O 表示不关注的字（实际上，在词性、专名联合标注中，不存在 O ）。示例如下：

```text
除\002了\002他\002续\002任\002十\002二\002届\002政\002协\002委\002员\002,\002马\002化\002腾\002,\002雷\002军\002,\002李\002彦\002宏\002也\002被\002推\002选\002为\002新\002一\002届\002全\002国\002人\002大\002代\002表\002或\002全\002国\002政\002协\002委\002员	p-B\002p-I\002r-B\002v-B\002v-I\002m-B\002m-I\002m-I\002ORG-B\002ORG-I\002n-B\002n-I\002w-B\002PER-B\002PER-I\002PER-I\002w-B\002PER-B\002PER-I\002w-B\002PER-B\002PER-I\002PER-I\002d-B\002p-B\002v-B\002v-I\002v-B\002a-B\002m-B\002m-I\002ORG-B\002ORG-I\002ORG-I\002ORG-I\002n-B\002n-I\002c-B\002n-B\002n-I\002ORG-B\002ORG-I\002n-B\002n-I
```

+ 我们随同代码一并发布了完全版的模型和相关的依赖数据。但是，由于模型的训练数据过于庞大，我们没有发布训练数据，仅在`data`目录下放置少数样本用以示例输入数据格式。

+ 模型依赖数据包括：
    1. 输入文本的词典，在`conf`目录下，对应`word.dic`
    2. 对输入文本中特殊字符进行转换的字典，在`conf`目录下，对应`q2b.dic`
    3. 标记标签的词典,在`conf`目录下，对应`tag.dic`

+ 在训练和预测阶段，我们都需要进行原始数据的预处理，具体处理工作包括：

    1. 从原始数据文件中抽取出句子和标签，构造句子序列和标签序列
    2. 将句子序列中的特殊字符进行转换
    3. 依据词典获取词对应的整数索引
    
### 代码结构说明
```text
.
├── README.md                           # 本文档
├── conf/                               # 词典目录
├── data/                               # 存放数据集的目录
├── downloads.sh                        # 用于下载数据和模型的脚本
├── gru-crf-model.png                   # README 用到的模型图片
├── reader.py                           # 文件读取相关函数
├── run_ernie_sequence_labeling.py      # 用于 finetune ERNIE 的代码
├── run_ernie.sh                        # 启用上面代码的脚本
├── run_sequence_labeling.py            # 词法分析任务代码
├── run.sh                              # 启用上面代码的脚本
└── utils.py                            # 常用工具函数
```

### 如何组建自己的模型

可以根据自己的需求，组建自定义的模型，具体方法如下所示：

1. 定义自己的词法分析模型，可以在 `../models/sequence_labeling/nets.py` 中添加自己的网络结构，具体接口要求可以参考默认的 `lex_net` 函数。
2. 模型训练、评估、预测的逻辑，需要在 `run.sh` 和 `run_ernie.sh` 中修改对应的模型路径、数据路径和词典路径等参数。

## 4. 其他
### 在论文中引用 LAC

如果您的学术工作成果中使用了 LAC，请您增加下述引用。我们非常欣慰 LAC 能够对您的学术工作带来帮助。

```text
@article{jiao2018LAC,
	title={Chinese Lexical Analysis with Deep Bi-GRU-CRF Network},
	author={Jiao, Zhenyu and Sun, Shuqi and Sun, Ke},
	journal={arXiv preprint arXiv:1807.01882},
	year={2018},
	url={https://arxiv.org/abs/1807.01882}
}
```
### 如何贡献代码
如果你可以修复某个 issue 或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度 进行打分（0-5分，越高越好）。如果你累计获得了 10 分，可以联系我们获得面试机会或为你写推荐信。
