# NCF

models/PaddleRec只是提供了经典推荐算法的Paddle实现，我们已经开源了功能更强大的工具组件[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 打通了推荐算法+分布式训练全流程，并提供了高级API，在单机和分布式间可以实现无缝切换。后续我们将在[PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec) Repo中发布新的模型和功能，models/PaddleRec不再更新维护。


## 简介

很多应用场景，并没有显性反馈的存在。因为大部分用户是沉默的用户，并不会明确给系统反馈“我对这个物品的偏好值是多少”。因此，推荐系统可以根据大量的隐性反馈来推断用户的偏好值。[《Neural Collaborative Filtering 》](https://arxiv.org/pdf/1708.05031.pdf)作者利用深度学习来对user和item特征进行建模，使模型具有非线性表达能力。具体来说使用多层感知机来学习user-item交互函数，提出了一种隐性反馈协同过滤解决方案。
