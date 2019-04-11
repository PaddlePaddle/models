## 简介

情感倾向分析（Sentiment Classification，简称Senta）针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极、中性。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有利的决策支持。可通过[AI开放平台-情感倾向性分析](http://ai.baidu.com/tech/nlp_apply/sentiment_classify) 线上体验。

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

下载经过预处理的数据，运行该脚本之后，data目录下会存在训练数据（train.tsv）、开发集数据（dev.tsv）、测试集数据（test.tsv）以及对应的词典（word_dict.txt）
```shell
cd data
sh download_data.sh
```

#### 模型下载

我们开源了基于ChnSentiCorp数据训练的情感倾向性分类模型（基于BOW、CNN、LSTM、ERNIE多种模型训练），可供用户直接使用
```shell
cd model
sh download_model.sh
```
#### 模型评估

基于上面的预训练模型和数据，可以运行下面的命令进行测试，查看预训练模型在开发集（dev.tsv）上的评测效果
```shell
# BOW、CNN、LSTM、BI-LSTM、GRU模型
sh run.sh eval
# ERNIE、ERNIE+BI-LSTM模型
sh run_ernie.sh eval
```

#### 模型训练

基于示例的数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证
```shell
# BOW、CNN、LSTM、BI-LSTM、GRU模型
sh run.sh train
# ERNIE、ERNIE+BI-LSTM模型
sh run_ernie.sh train
```
训练完成后，可修改```run.sh```中init_checkpoint参数，进行模型评估和预测

#### 模型预测

利用已有模型，可以运行下面命令，对未知label的数据（test.tsv）进行预测
```shell
# BOW、CNN、LSTM、BI-LSTM、GRU模型
sh run.sh infer
#ERNIE+BI-LSTM模型
sh run_ernie.sh infer
```

## 进阶使用

#### 任务定义

传统的情感分类主要基于词典或者特征工程的方式进行分类，这种方法需要繁琐的人工特征设计和先验知识，理解停留于浅层并且扩展泛化能力差。为了避免传统方法的局限，我们采用近年来飞速发展的深度学习技术。基于深度学习的情感分类不依赖于人工特征，它能够端到端的对输入文本进行语义理解，并基于语义表示进行情感倾向的判断。
#### 模型原理介绍

本项目针对情感倾向性分类问题，开源了一系列模型，供用户可配置地使用：

+ BOW（Bag Of Words）模型，是一个非序列模型，使用基本的全连接结构；
+ CNN（Convolutional Neural Networks），是一个基础的序列模型，能处理变长序列输入，提取局部区域之内的特征；
+ GRU（Gated Recurrent Unit），序列模型，能够较好地解决序列文本中长距离依赖的问题；
+ LSTM（Long Short Term Memory），序列模型，能够较好地解决序列文本中长距离依赖的问题；
+ BI-LSTM（Bidirectional Long Short Term Memory），序列模型，采用双向LSTM结构，更好地捕获句子中的语义特征；
+ ERNIE（Enhanced Representation through kNowledge IntEgration），百度自研基于海量数据和先验知识训练的通用文本语义表示模型，并基于此在情感倾向分类数据集上进行fine-tune获得。
+ ERNIE+BI-LSTM，基于ERNIE语义表示对接上层BI-LSTM模型，并基于此在情感倾向分类数据集上进行Fine-tune获得；

#### 数据格式说明

训练、预测、评估使用的数据可以由用户根据实际的应用场景，自己组织数据。数据由两列组成，以制表符分隔，第一列是以空格分词的中文文本（分词预处理方法将在下文具体说明），文件为utf8编码；第二列是情感倾向分类的类别（0表示消极；1表示积极），注意数据文件第一行固定表示为"text_a\tlabel"

```text
特 喜欢 这种 好看的 狗狗	              1
这 真是 惊艳 世界 的 中国 黑科技	      1
环境 特别 差 ，脏兮兮 的，再也 不去 了      0
```
注：本项目额外提供了分词预处理脚本（在本项目的preprocess目录下），可供用户使用，具体使用方法如下：
```shell
python tokenizer.py --test_data_dir ./test.txt.utf8 --batch_size 1 > test.txt.utf8.seg

#其中test.txt.utf8为待分词的文件，一条文本数据一行，utf8编码，分词结果存放在test.txt.utf8.seg文件中。
```

#### 代码结构说明

```text
.
├── senta_config.json       # 模型配置文件
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
在 ```senta_config.json``` 中需要将 ```model_type``` 改为用户自定义的 ```user_net```
3. 模型训练，运行训练、评估、预测脚本即可（具体方法同上）

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
