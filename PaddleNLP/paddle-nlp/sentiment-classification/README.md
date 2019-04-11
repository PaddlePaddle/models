## 简介

情感倾向分析（Sentiment Classification，简称Senta）针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极、中性。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有利的决策支持。可通过[AI开放平台-对话情绪识别](http://ai.baidu.com/tech/nlp_apply/sentiment_classify) 线上体验。

情感是人类的一种高级智能行为，为了识别文本的情感倾向，需要深入的语义建模。另外，不同领域（如餐饮、体育）在情感的表达各不相同，因而需要有大规模覆盖各个领域的数据进行模型训练。为此，我们通过基于深度学习的语义模型和大规模数据挖掘解决上述两个问题。效果上，我们基于开源情感倾向分类数据集ChnSentiCorp进行评测；此外，我们还开源了百度基于海量数据训练好的模型，该模型在ChnSentiCorp数据集上fine-tune之后，可以得到更好的效果。具体数据如下所示：
 
| 模型 | dev | test | 模型（finetune） |dev | test | 
| :------| :------ | :------ | :------ |:------ | :------
| BOW | 90.3% | 90.7% | BOW |91.3% | 90.6% |
| CNN | 90.3% | 90.1% | CNN |92.4% | 91.8% |
| LSTM | 91.0% | 90.9% | LSTM |93.3% | 92.2% |
| GRU | 91.0% | 90.7% | GRU |93.3% | 93.2% |
| BI-LSTM | 90.5% | 90.8% | BI-LSTM |92.8% | 91.4% |
| ERNIE | 94.5% | 95.8% | ERNIE |94.8% | 95.3% |
| ERNIE+BI-LSTM | 95.1% | 95.4% | ERNIE+BI-LSTM |95.2% | 95.8% |



## 快速开始

本项目依赖于 Paddlepaddle Fluid 1.3.2，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

#### 数据准备

下载经过预处理的数据，运行该脚本之后，会生成data目录，data目录下有训练集数据（train.tsv）、开发集数据（dev.tsv）、测试集数据（test.tsv）、 待预测数据（infer.tsv）以及对应词典（vocab.txt）
```shell
sh download_data.sh
```

#### 模型下载

我们开源了基于海量数据训练好的对话情绪识别模型（基于TextCNN模型训练），可供用户直接使用，运行脚本后，会生成models目录，models目录下会有预训练的模型文件
```shell
sh download_model.sh
```
#### 模型评估

基于已有的预训练模型和数据，可以运行下面的命令进行测试，查看预训练的模型在测试集（test.tsv）上的评测结果
```shell
sh run.sh eval
```

#### 模型训练

基于示例的数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证
```shell
sh run.sh train
```
训练完成后，可修改```run.sh```中init_checkpoint参数，进行模型评估和预测

#### 模型预测

基于预训练模型，可在新的数据集（infer.tsv）上进行预测，得到模型预测结果及概率
```shell
sh run.sh infer
```

## 进阶使用

#### 任务定义

对话情绪识别任务输入是一段用户文本，输出是检测到的情绪类别，包括消极、积极、中性，这是一个经典的短文本三分类任务。

#### 模型原理介绍

本项目针对对话情绪识别问题，开源了一系列分类模型，供用户可配置地使用：

+ BOW：Bag Of Words，是一个非序列模型，使用基本的全连接结构；
+ CNN：浅层CNN模型，能够处理变长的序列输入，提取一个局部区域之内的特征；；
+ TextCNN：多卷积核CNN模型，能够更好地捕捉句子局部相关性；
+ LSTM：单层LSTM模型，能够较好地解决序列文本中长距离依赖的问题；
+ BI-LSTM：双向单层LSTM模型，采用双向LSTM结构，更好地捕获句子中的语义特征；
+ ERNIE：百度自研基于海量数据和先验知识训练的通用文本语义表示模型，并基于此在对话情绪分类数据集上进行fine-tune获得。

#### 数据格式说明

训练、预测、评估使用的数据示例如下，数据由两列组成，以制表符（'\t'）分隔，第一列是情绪分类的类别（0表示消极；1表示中性；2表示积极），第二列是以空格分词的中文文本，文件为utf8编码。

```text
0   谁 骂人 了 ？ 我 从来 不 骂人 ， 我 骂 的 都 不是 人 ， 你 是 人 吗 ？
1   我 有事 等会儿 就 回来 和 你 聊
2   我 见到 你 很高兴 谢谢 你 帮 我
```
注：本项目额外提供了分词预处理脚本（在preprocess目录下），可供用户使用，具体使用方法如下：
```shell
python tokenizer.py --test_data_dir ./test.txt.utf8 --batch_size 1 > test.txt.utf8.seg
```

#### 代码结构说明

```text
.
├── config.json             # 模型配置文件
├── config.py               # 定义了该项目模型的相关配置，包括具体模型类别、以及模型的超参数
├── reader.py               # 定义了读入数据，加载词典的功能
├── run_classifier.py       # 该项目的主函数，封装包括训练、预测、评估的部分
├── run_ernie_classifier.py # 基于ERNIE表示的项目的主函数
├── run_ernie.sh            # 基于ERNIE的训练、预测、评估运行脚本
├── run.sh                  # 训练、预测、评估运行脚本
├── utils.py                # 定义了其他常用的功能函数
```

#### 如何组建自己的模型

可以根据自己的需求，组建自定义的模型，具体方法如下所示：

1. 定义自己的网络结构 
用户可以在 ```models/classify/nets.py``` 中，定义自己的模型，只需要增加新的函数即可。假设用户自定义的函数名为```user_net```
2. 更改模型配置
在 ```config.json``` 中需要将 ```model_type``` 改为用户自定义的 ```user_net```
3. 模型训练，运行训练、评估、预测脚本即可（具体方法同上）

#### 使用ERNIE进行finetune

1. 下载 ERNIE 预训练模型
```
mkdir -p models/ernie
cd models/ernie
wget --no-check-certificate https://ernie.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz
tar xvf ERNIE_stable-1.0.1.tar.gz
rm ERNIE_stable-1.0.1.tar.gz
```
2. 配置 ERNIE 模型及数据
通过 ```run_ernie.sh``` 配置ERNIE模型路径及数据路径，例如
```
MODEL_PATH=./models/ernie
TASK_DATA_PATH=./data
```
3. 模型训练
```
sh run_ernie.sh train
```
训练、评估、预测详细配置，请查看 ```run_ernie.sh```

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
