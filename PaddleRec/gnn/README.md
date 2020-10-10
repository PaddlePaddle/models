# SR-GNN

models/PaddleRec只是提供了经典推荐算法的Paddle实现，我们已经开源了功能更强大的工具组件[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 打通了推荐算法+分布式训练全流程，并提供了高级API，在单机和分布式间可以实现无缝切换。后续我们将在[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) Repo中发布新的模型和功能，models/PaddleRec不再更新维护。

## 简介

SR-GNN模型的介绍可以参阅论文[Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)。

本文解决的是Session-based Recommendation这一问题,过程大致分为以下四步：

是对所有的session序列通过有向图进行建模。

然后通过GNN，学习每个node（item）的隐向量表示

然后通过一个attention架构模型得到每个session的embedding

最后通过一个softmax层进行全表预测

我们复现了论文效果，在DIGINETICA数据集上P@20可以达到50.7
