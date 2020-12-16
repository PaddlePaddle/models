# listwise

models/PaddleRec只是提供了经典推荐算法的Paddle实现，我们已经开源了功能更强大的工具组件[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 打通了推荐算法+分布式训练全流程，并提供了高级API，在单机和分布式间可以实现无缝切换。后续我们将在[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) Repo中发布新的模型和功能，models/PaddleRec不再更新维护。

## 简介

[《Sequential Evaluation and Generation Framework for Combinatorial Recommender System》]( https://arxiv.org/pdf/1902.00245.pdf)是百度2019年发布的推荐系统融合模型，用于优化推荐序列的整体性能（如总点击），该模型由Generator和Evaluator两部分组成，Generator负责生成若干个候选序列，Evaluator负责从候选序列中筛选出最好的序列推荐给用户，达到最大化序列整体性能的目的。

本项目在paddlepaddle上实现该融合模型的Evaluator部分，构造数据集验证模型的正确性。
