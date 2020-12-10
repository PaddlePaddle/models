# DIN

models/PaddleRec只是提供了经典推荐算法的Paddle实现，我们已经开源了功能更强大的工具组件[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 打通了推荐算法+分布式训练全流程，并提供了高级API，在单机和分布式间可以实现无缝切换。后续我们将在[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) Repo中发布新的模型和功能，models/PaddleRec不再更新维护。

## 简介

DIN模型的介绍可以参阅论文[Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)。

DIN通过一个兴趣激活模块(Activation Unit)，用预估目标Candidate ADs的信息去激活用户的历史点击商品，以此提取用户与当前预估目标相关的兴趣。

权重高的历史行为表明这部分兴趣和当前广告相关，权重低的则是和广告无关的”兴趣噪声“。我们通过将激活的商品和激活权重相乘，然后累加起来作为当前预估目标ADs相关的兴趣状态表达。

最后我们将这相关的用户兴趣表达、用户静态特征和上下文相关特征，以及ad相关的特征拼接起来，输入到后续的多层DNN网络，最后预测得到用户对当前目标ADs的点击概率。
