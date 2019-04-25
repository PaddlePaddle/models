DuReader是一个端到端的机器阅读理解神经网络模型，能够在给定文档和问题的情况下，定位文档中问题的答案。我们首先利用双向注意力网络获得文档和问题的相同向量空间的表示，然后使用`point network` 定位文档中答案的位置。实验显示，我们的模型能够获得在Dureader数据集上SOTA的结果。

# 算法介绍
DuReader模型主要实现了论文[BiDAF](https://arxiv.org/abs/1611.01603)， [Match-LSTM](https://arxiv.org/abs/1608.07905)中的模型结构。

模型在层次上可以分为5层：

- **词嵌入层** 将每一个词映射到一个向量(这个向量可以是预训练好的)。
- **编码层** 使用双向LSMT网络获得问题和文档的每一个词的上下文信息。
- **注意力层** 通过双向注意力网络获得文档的问题向量空间表示。更多参考[BiDAF](https://arxiv.org/abs/1611.01603)。
- **融合层** 通过双向LSTM网络获得文档向量表示中上下文的关联性信息。
- **输出层** 通过`point network`预测答案在问题中的位置。更多参考 [Match-LSTM](https://arxiv.org/abs/1608.07905)。

## 数据准备
### 下载数据集
通过如下脚本下载数据集:
```
cd data && bash download.sh
```
模型默认使用DuReader数据集，是百度开源的真实阅读理解数据，更多参考[DuReader Dataset Homepage](https://ai.baidu.com//broad/subordinate?dataset=dureader)

### 下载第三方依赖
我们使用Bleu和Rouge作为度量指标， 这些度量指标的源码位于[coco-caption](https://github.com/tylin/coco-caption)， 可以使用如下命令下载源码:

```
cd utils && bash download_thirdparty.sh
```
### 环境依赖
当前模型是在paddlepaddle 1.2版本上测试， 因此建议在1.2版本上使用本模型。关于PaddlePaddle的安装可以参考[PaddlePaddle Homepage](http://paddlepaddle.org)。

## 模型训练
### 段落抽取
在段落抽取阶段，主要是使用文档相关性score对文档内容进行优化， 抽取的结果将会放到`data/extracted/`目录下。如果你用demo数据测试，可以跳过这一步。如果你用dureader数据，需要指定抽取的数据目录，命令如下：
```
bash run.sh --para_extraction --trainset data/preprocessed/trainset/zhidao.train.json data/preprocessed/trainset/search.train.json --devset data/preprocessed/devset/zhidao.dev.json data/preprocessed/devset/search.dev.json --testset data/preprocessed/testset/zhidao.test.json data/preprocessed/testset/search.test.json
```
其中参数 `trainset`/`devset`/`testset`分别对应训练、验证和测试数据集(下同)。
### 词典准备
在训练模型之前，我们应该确保数据已经准备好。在准备阶段，通过全部数据文件生成一个词典，这个词典会在后续的训练和预测中用到。你可以通过如下命令生成词典：
```
bash run.sh --prepare
```
上面的命令默认使用demo数据，如果想使用dureader数据集，应该按照如下方式指定：
```
bash run.sh --prepare --trainset data/extracted/trainset/zhidao.train.json data/extracted/trainset/search.train.json --devset data/extracted/devset/zhidao.dev.json data/extracted/devset/search.dev.json --testset data/extracted/testset/zhidao.test.json data/extracted/testset/search.test.json
```
其中参数 `trainset`/`devset`/`testset`分别对应训练、验证和测试数据集。
### 模型训练
训练模型的启动命令如下：
```
bash run.sh --train
```
上面的命令默认使用demo数据，如果想使用dureader数据集，应该按照如下方式指定：
```
bash run.sh --train --trainset data/extracted/trainset/zhidao.train.json data/extracted/trainset/search.train.json --devset data/extracted/devset/zhidao.dev.json data/extracted/devset/search.dev.json --testset data/extracted/testset/zhidao.test.json data/extracted/testset/search.test.json
```
可以通过设置超参数更改训练的配置，比如通过`--learning_rate NUM`更改学习率，通过`--pass_num NUM`更改训练的轮数
训练的过程中，每隔一定迭代周期，会测试在验证集上的性能指标， 通过`--dev_interval NUM`设置周期大小

### 模型评测
在模型训练结束后，如果想使用训练好的模型进行评测，获得度量指标，可以使用如下命令:
```
bash run.sh --evaluate  --load_dir data/models/1
```
其中，`--load_dir data/models/1`是模型的checkpoint目录

上面的命令默认使用demo数据，如果想使用dureader数据集，应该按照如下方式指定：
```
bash run.sh --evaluate  --load_dir data/models/1  --devset data/extracted/devset/zhidao.dev.json data/extracted/devset/search.dev.json --testset data/extracted/testset/zhidao.test.json data/extracted/testset/search.test.json
```

### 预测
使用训练好的模型，对问答文档数据直接预测结果，获得答案，可以使用如下命令:
```
bash run.sh --predict --load_dir data/models/1
```
上面的命令默认使用demo数据，如果想使用dureader数据集，应该按照如下方式指定：
```
bash run.sh --predict --load_dir data/models/1 --testset data/extracted/testset/search.test.json data/extracted/testset/zhidao.test.json
```
其中`--testset`指定了预测用的数据集，生成的问题答案默认会放到`data/results/` 目录，你可以通过参数`--result_dir DIR_PATH`更改配置

### 实验结果
验证集 ROUGE-L:47.65。

这是在P40上，使用4卡GPU，batch size=4*32的训练5个epoch(约30个小时)的结果，如果使用单卡，指标可能会略有降低，但在验证集上的ROUGE-L也不小于47。

## 参考文献
[Machine Comprehension Using Match-LSTM and Answer Pointer](https://arxiv.org/abs/1608.07905)

[Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)
