
# 基于DNN模型的点击率预估模型

models/PaddleRec只是提供了经典推荐算法的Paddle实现，我们已经开源了功能更强大的工具组件[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 打通了推荐算法+分布式训练全流程，并提供了高级API，在单机和分布式间可以实现无缝切换。后续我们将在[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) Repo中发布新的模型和功能，models/PaddleRec不再更新维护。

## 介绍
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。本模型实现了下述论文中提出的DNN模型：

