# MMoE

##简介

MMoE是经典的多任务（multi-task）模型，原论文[Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-) 发表于KDD 2018.

多任务模型通过学习不同任务的联系和差异，可提高每个任务的学习效率和质量。多任务学习的的框架广泛采用shared-bottom的结构，不同任务间共用底部的隐层。这种结构本质上可以减少过拟合的风险，但是效果上可能受到任务差异和数据分布带来的影响。论文中提出了一个Multi-gate Mixture-of-Experts(MMoE)的多任务学习结构。MMoE模型刻画了任务相关性，基于共享表示来学习特定任务的函数，避免了明显增加参数的缺点。(https://zhuanlan.zhihu.com/p/55752344)

我们基于实际工业界场景实现了MMoE的核心思想。

## 配置
1.6 及以上

## 数据

我们采用了随机数据作为训练数据，可以根据自己的数据调整data部分。

## 训练

```
python mmoe_train.py
```

# 未来工作

1. 添加预测部分

2. 添加公开数据集的结果
