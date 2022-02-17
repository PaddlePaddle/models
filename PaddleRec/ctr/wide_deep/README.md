# wide&deep

models/PaddleRec只是提供了经典推荐算法的Paddle实现，我们已经开源了功能更强大的工具组件[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 打通了推荐算法+分布式训练全流程，并提供了高级API，在单机和分布式间可以实现无缝切换。后续我们将在[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) Repo中发布新的模型和功能，models/PaddleRec不再更新维护。

## 简介

[《Wide & Deep Learning for Recommender Systems》]( https://arxiv.org/pdf/1606.07792.pdf)是Google 2016年发布的推荐框架，wide&deep设计了一种融合浅层（wide）模型和深层（deep）模型进行联合训练的框架，综合利用浅层模型的记忆能力和深层模型的泛化能力，实现单模型对推荐系统准确性和扩展性的兼顾。从推荐效果和服务性能两方面进行评价：

1. 效果上，在Google Play 进行线上A/B实验，wide&deep模型相比高度优化的Wide浅层模型，app下载率+3.9%。相比deep模型也有一定提升。
2. 性能上，通过切分一次请求需要处理的app 的Batch size为更小的size，并利用多线程并行请求达到提高处理效率的目的。单次响应耗时从31ms下降到14ms。
