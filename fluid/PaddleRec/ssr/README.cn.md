# 序列语义召回模型

## 介绍
在新闻推荐领域，与传统的推荐电影和音乐的娱乐推荐系统不同，新闻推荐场景中有很多新的问题要解决。
- 用户画像特征存在非常稀疏的情况，例如一个用户可能经常在一个新闻推荐的app匿名登录，或者一个用户可能会阅读一篇全新的文章。
- 与电影、音乐等资源相比，新闻的产生和消亡速度非常快。通常，每天会有上千篇新闻产生。而新闻的消化速度快，主要原因是因为人们更关心最新发生的事情。
- 用户的兴趣在新闻推荐场景中会经常发生变化。新闻的内容本身会很大程度影响用户的阅读行为，即使新闻的类别并不属于用户的长期兴趣中的一种。在新闻推荐的场景下，用户的阅读行为是由用户的短期兴趣和长期兴趣共同决定的。


[GRU4Rec](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/gru4rec) 利用Gated-Recurrent-Unit对用户的短期兴趣和长期兴趣做了很好的建模。Recurrent Neural Network本身的泛化能力刻画了用户阅读行为的序列相似度，这种泛化能力从一定程度上缓解了用户画像特征稀疏的问题。
然而，GRU4Rec模型是对一个封闭的新闻集合进行预测的，把推荐新闻的问题当成了分类问题，但在新闻推荐场景中，每天新闻都是动态产生和消亡的，不适合用分类的方法进行解决。

序列语义召回模型（Sequence Semantic Retrieval-SSR）与Multi-Rate Deep Learning for Temporal Recommendation, SIGIR 2016提出的方法类似，针对GRU4Rec把新闻推荐问题当做分类问题解决而不能扩展到领域外资源的问题提出。序列语义召回模型分为两个部分，一个是兴趣匹配模块，一个是检索模块。
- SSR的想法是对用户的兴趣利用匹配的网络结构进行建模，一篇新闻表示可以利用新闻相关的一些特征进行描述，这样在线推荐的阶段可以扩展到训练集合以外的新闻资源。
- 有了新闻的表示，我们能够通过一些向量索引的方法在线进行新闻推荐，这是SSR的检索部分。

## 数据集
数据预处理的方法参考[GRU4Rec项目](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleRec/gru4rec)。注意这里需要复用GRU4Rec项目中脚本进行预处理。

## 训练
在开始之前，设置PYTHONPATH环境
```
export PYTHONPATH=./models/fluid:$PYTHONPATH
```

训练的命令行提示可以通过使用命令`python train.py -h`获得
``` bash
python train.py --train_file rsc15_train_tr_paddle.txt
```

## 创建索引
TBA

## 检索阶段
TBA
