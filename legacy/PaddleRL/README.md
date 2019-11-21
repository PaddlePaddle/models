PaddleRL
============

强化学习
--------

强化学习是近年来一个愈发重要的机器学习方向，特别是与深度学习相结合而形成的深度强化学习(Deep Reinforcement Learning, DRL)，取得了很多令人惊异的成就。人们所熟知的战胜人类顶级围棋职业选手的 AlphaGo 就是 DRL 应用的一个典型例子，除游戏领域外，其它的应用还包括机器人、自然语言处理等。

深度强化学习的开山之作是在Atari视频游戏中的成功应用， 其可直接接受视频帧这种高维输入并根据图像内容端到端地预测下一步的动作，所用到的模型被称为深度Q网络(Deep Q-Network, DQN)。本实例就是利用PaddlePaddle Fluid这个灵活的框架，实现了 DQN 及其变体，并测试了它们在 Atari 游戏中的表现。

-  [DeepQNetwork](https://github.com/PaddlePaddle/models/blob/develop/PaddleRL/DeepQNetwork/README_cn.md)
