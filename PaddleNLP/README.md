## 简介

PaddleNLP是百度开源的工业级NLP工具与预训练模型集，能够适应全面丰富的NLP任务，方便开发者灵活插拔尝试多种网络结构，并且让应用最快速达到工业级效果。

PaddleNLP完全基于[Paddle Fluid](http://www.paddlepaddle.org/)开发，并提供依托于百度百亿级大数据的预训练模型，能够极大地方便NLP研究者和工程师快速应用。使用者可以用Paddle NLP快速实现文本分类、文本匹配、序列标注、阅读理解、智能对话等NLP任务的组网、建模和部署，而且可以直接使用百度开源工业级预训练模型进行快速应用。用户在极大地减少研究和开发成本的同时，也可以获得更好的基于工业实践的应用效果。

PaddleNLP的特点与优势：
1. 全面丰富的中文NLP应用任务。
2. 任务与网络解耦，网络灵活可插拔。
3. 强大的工业化预训练模型，打造优异应用效果。

#### 目录结构
```text
.
├── dialogue_model_toolkit            # 对话模型工具箱
├── emotion_detection                 # 对话情绪识别
├── knowledge_driven_dialogue         # 知识驱动对话
├── language_model                    # 语言模型
├── language_representations_kit      # 语言表示工具箱
├── lexical_analysis                  # 词法分析
├── models                            # 共享网络
│   ├── __init__.py
│   ├── classification
│   ├── dialogue_model_toolkit
│   ├── language_model
│   ├── matching
│   ├── neural_machine_translation
│   ├── reading_comprehension
│   ├── representation
│   ├── sequence_labeling
│   └── transformer_encoder.py
├── neural_machine_translation        # 机器翻译
├── preprocess                        # 共享文本预处理工具
│   ├── __init__.py
│   ├── ernie
│   ├── padding.py
│   └── tokenizer
├── reading_comprehension             # 阅读理解
├── sentiment_classification          # 文本情感分析
├── similarity_net                    # 短文本语义匹配
```
除了models和preprocess分别是共享组网集与共享预处理流程，其他路径都是相互独立的任务。可以直接进入各任务路径中运行任务。

以下以情感分析任务为例。
## 快速开始
#### 版本依赖
本项目依赖于 Paddle Fluid 1.3.2，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

python版本依赖python 2.7

#### 数据准备

下载经过预处理的数据，运行该脚本之后，data目录下会存在训练数据（train.tsv）、开发集数据（dev.tsv）、测试集数据（test.tsv）以及对应的词典（word_dict.txt）
```shell
wget https://baidu-nlp.bj.bcebos.com/Senta_data.tar
tar -xvf Senta_data.tar 
```

#### 模型下载
我们开源了基于ChnSentiCorp数据训练的情感倾向性分类模型（基于BOW、CNN、LSTM、ERNIE多种模型训练），可供用户直接使用。我们提供两种下载方式。

**方式一**：基于PaddleHub命令行工具（PaddleHub[安装方式](https://github.com/PaddlePaddle/PaddleHub)）
```shell
hub download sentiment_classification --output_path ./
tar -xvf sentiment_classification-1.0.0.tar.gz
```

**方式二**：直接下载
```shell
wget https://baidu-nlp.bj.bcebos.com/sentiment_classification-1.0.0.tar.gz
tar -xvf sentiment_classification-1.0.0.tar.gz
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
环境 特别 差 ，脏兮兮 的，再也 不去 了     0
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
用户可以在 ```models/classification/nets.py```中，定义自己的模型，只需要增加新的函数即可。假设用户自定义的函数名为```user_net```
2. 更改模型配置
在 ```senta_config.json```中需要将 ```model_type```改为用户自定义的 ```user_net```
3. 模型训练，运行训练、评估、预测脚本即可（具体方法同上）

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
