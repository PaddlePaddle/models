## 简介


情感是人类的一种高级智能行为，为了识别文本的情感倾向，需要深入的语义建模。另外，不同领域（如餐饮、体育）在情感的表达各不相同，因而需要有大规模覆盖各个领域的数据进行模型训练。为此，我们通过基于深度学习的语义模型和大规模数据挖掘解决上述两个问题。效果上，我们基于开源情感倾向分类数据集ChnSentiCorp进行评测；此外，我们还开源了百度基于海量数据训练好的模型，该模型在ChnSentiCorp数据集上fine-tune之后（基于开源模型进行Finetune的方法请见下面章节），可以得到更好的效果。具体数据如下所示：

| 模型 | dev | 
| :------| :------ |
| CNN | 90.6% |


## 快速开始

本项目依赖于 Paddlepaddle 1.3.2 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

python版本依赖python 2.7

#### 安装代码

克隆数据集代码库到本地
```shell
git clone https://github.com/PaddlePaddle/models.git
cd models/PaddleNLP/sentiment_classification
```

#### 数据准备

下载经过预处理的数据，文件解压之后，senta_data目录下会存在训练数据（train.tsv）、开发集数据（dev.tsv）、测试集数据（test.tsv）以及对应的词典（word_dict.txt）
```shell
wget https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
tar -zxvf sentiment_classification-dataset-1.0.0.tar.gz
```

#### 模型训练

基于示例的数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证
```shell
python train.py
```

#### 模型预测

利用已有模型，可以运行下面命令，对未知label的数据（test.tsv）进行预测
```shell
python infer.py
```

## 进阶使用

#### 任务定义

传统的情感分类主要基于词典或者特征工程的方式进行分类，这种方法需要繁琐的人工特征设计和先验知识，理解停留于浅层并且扩展泛化能力差。为了避免传统方法的局限，我们采用近年来飞速发展的深度学习技术。基于深度学习的情感分类不依赖于人工特征，它能够端到端的对输入文本进行语义理解，并基于语义表示进行情感倾向的判断。
#### 模型原理介绍

本项目针对情感倾向性分类问题，：

+ CNN（Convolutional Neural Networks），是一个基础的序列模型，能处理变长序列输入，提取局部区域之内的特征；

#### 数据格式说明

训练、预测、评估使用的数据可以由用户根据实际的应用场景，自己组织数据。数据由两列组成，以制表符分隔，第一列是以空格分词的中文文本（分词预处理方法将在下文具体说明），文件为utf8编码；第二列是情感倾向分类的类别（0表示消极；1表示积极），注意数据文件第一行固定表示为"text_a\tlabel"

```text
特 喜欢 这种 好看的 狗狗	              1
这 真是 惊艳 世界 的 中国 黑科技	      1
环境 特别 差 ，脏兮兮 的，再也 不去 了     0
```

#### 代码结构说明

```text
.
├── reader.py               # 定义了读入数据，加载词典的功能
├── train.py                # 该项目的主函数，封装包括训练、预测、评估的部分
├── nets.py                 # 网络结构
├── utils.py                # 定义了其他常用的功能函数
```

