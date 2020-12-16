# GRU4REC

models/PaddleRec只是提供了经典推荐算法的Paddle实现，我们已经开源了功能更强大的工具组件[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 打通了推荐算法+分布式训练全流程，并提供了高级API，在单机和分布式间可以实现无缝切换。后续我们将在[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) Repo中发布新的模型和功能，models/PaddleRec不再更新维护。


## 简介

GRU4REC模型的介绍可以参阅论文[Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)。

论文的贡献在于首次将RNN（GRU）运用于session-based推荐，相比传统的KNN和矩阵分解，效果有明显的提升。

论文的核心思想是在一个session中，用户点击一系列item的行为看做一个序列，用来训练RNN模型。预测阶段，给定已知的点击序列作为输入，预测下一个可能点击的item。

session-based推荐应用场景非常广泛，比如用户的商品浏览、新闻点击、地点签到等序列数据。

支持三种形式的损失函数, 分别是全词表的cross-entropy, 负采样的Bayesian Pairwise Ranking和负采样的Cross-entropy.

我们基本复现了论文效果，recall@20的效果分别为
