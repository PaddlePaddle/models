# 个性化推荐中的多视角Simnet模型

models/PaddleRec只是提供了经典推荐算法的Paddle实现，我们已经开源了功能更强大的工具组件[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 打通了推荐算法+分布式训练全流程，并提供了高级API，在单机和分布式间可以实现无缝切换。后续我们将在[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) Repo中发布新的模型和功能，models/PaddleRec不再更新维护。

## 介绍
在个性化推荐场景中，推荐系统给用户提供的项目（Item）列表通常是通过个性化的匹配模型计算出来的。在现实世界中，一个用户可能有很多个视角的特征，比如用户Id，年龄，项目的点击历史等。一个项目，举例来说，新闻资讯，也会有多种视角的特征比如新闻标题，新闻类别等。多视角Simnet模型是可以融合用户以及推荐项目的多个视角的特征并进行个性化匹配学习的一体化模型。这类模型在很多工业化的场景中都会被使用到，比如百度的Feed产品中。
