# youtube dnn

models/PaddleRec只是提供了经典推荐算法的Paddle实现，我们已经开源了功能更强大的工具组件[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 打通了推荐算法+分布式训练全流程，并提供了高级API，在单机和分布式间可以实现无缝切换。后续我们将在[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) Repo中发布新的模型和功能，models/PaddleRec不再更新维护。


## 简介

[《Deep Neural Networks for YouTube Recommendations》](https://link.zhihu.com/?target=https%3A//static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf) 这篇论文是google的YouTube团队在推荐系统上DNN方面的尝试，是经典的向量化召回模型，主要通过模型来学习用户和物品的兴趣向量，并通过内积来计算用户和物品之间的相似性，从而得到最终的候选集。YouTube采取了两层深度网络完成整个推荐过程：

1.第一层是**Candidate Generation Model**完成候选视频的快速筛选，这一步候选视频集合由百万降低到了百的量级。

2.第二层是用**Ranking Model**完成几百个候选视频的精排。

本项目在paddlepaddle上完成YouTube dnn的召回部分Candidate Generation Model，分别获得用户和物品的向量表示，从而后续可以通过其他方法（如用户和物品的余弦相似度）给用户推荐物品。
