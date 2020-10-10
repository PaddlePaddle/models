# DSSM

models/PaddleRec只是提供了经典推荐算法的Paddle实现，我们已经开源了功能更强大的工具组件[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 打通了推荐算法+分布式训练全流程，并提供了高级API，在单机和分布式间可以实现无缝切换。后续我们将在[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) Repo中发布新的模型和功能，models/PaddleRec不再更新维护。


## 简介

DSSM[《Learning Deep Structured Semantic Models for Web Search using Clickthrough Data》](  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf  )即基于深度网络的语义模型，其核心思想是将query和doc映射到共同维度的语义空间中，通过最大化query和doc语义向量之间的余弦相似度，从而训练得到隐含语义模型，达到检索的目的，并通过word hashing方法来减少输入向量的维度。DSSM有很广泛的应用，比如：搜索引擎检索，广告相关性，问答系统，机器翻译等。
