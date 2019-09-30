# 强化学习
本页将介绍如何使用PaddlePaddle在DyGraph模式下实现典型强化学习算法，包括[安装](#installation)、[训练](#training-a-model)、[输出](#log)、[模型评估](#evaluation)。

动态图文档请见[Dygraph](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/user_guides/howto/dygraph/DyGraph.html)

---
## 内容
- [安装](#installation)
- [训练](#training-a-model)
- [输出](#log)

## 安装

在当前目录下运行样例代码需要PadddlePaddle Fluid的v1.4.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据安装文档中的说明来更新PaddlePaddle。

除了paddle以外，请安装gym模拟器（https://gym.openai.com/ ），这个模拟器内包含了十分丰富的模拟环境，可以极大地便利强化学习研究。

## 训练
教程中使用`Cartpole`作为模拟环境（关于Cartpole: https://gym.openai.com/envs/CartPole-v0 ），并包含了2个典型的强化学习算法：reinforce.py 和 actor_critic.py，可以通过如下的方式启动训练：
```
env CUDA_VISIBLE_DEVICES=0 python reinforce.py
```
或
```
env CUDA_VISIBLE_DEVICES=0 python actor_critic.py
```

## 输出
执行训练开始后，将得到类似如下的输出。

```
env CUDA_VISIBLE_DEVICES=0 python reinforce.py

Episode 10      Last reward: 20.00      Average reward: 14.96
Episode 20      Last reward: 35.00      Average reward: 20.56
Episode 30      Last reward: 26.00      Average reward: 23.18
Episode 40      Last reward: 21.00      Average reward: 28.68
Episode 50      Last reward: 21.00      Average reward: 30.06
Episode 60      Last reward: 27.00      Average reward: 37.21
Episode 70      Last reward: 67.00      Average reward: 47.69
Episode 80      Last reward: 46.00      Average reward: 55.25
Episode 90      Last reward: 113.00     Average reward: 80.11
Episode 100     Last reward: 124.00     Average reward: 89.36
Episode 110     Last reward: 97.00      Average reward: 98.29
Episode 120     Last reward: 200.00     Average reward: 110.29
Episode 130     Last reward: 200.00     Average reward: 142.01
Episode 140     Last reward: 157.00     Average reward: 162.18
Episode 150     Last reward: 101.00     Average reward: 165.37
Episode 160     Last reward: 119.00     Average reward: 156.74
Episode 170     Last reward: 114.00     Average reward: 146.62
Episode 180     Last reward: 149.00     Average reward: 140.74
Episode 190     Last reward: 114.00     Average reward: 149.52
Episode 200     Last reward: 124.00     Average reward: 130.40
Episode 210     Last reward: 103.00     Average reward: 119.44
Episode 220     Last reward: 200.00     Average reward: 120.50
Episode 230     Last reward: 172.00     Average reward: 126.33
Episode 240     Last reward: 187.00     Average reward: 139.02
Episode 250     Last reward: 170.00     Average reward: 154.12
Episode 260     Last reward: 172.00     Average reward: 167.44
Episode 270     Last reward: 195.00     Average reward: 175.00
Episode 280     Last reward: 200.00     Average reward: 178.56
Episode 290     Last reward: 200.00     Average reward: 187.16
Episode 300     Last reward: 200.00     Average reward: 192.32
Solved! Running reward is now 195.156645521 and the last episode runs to 200 time steps!

```
或
```
env CUDA_VISIBLE_DEVICES=0 python actor_critic.py

Episode 10      Last reward: 131.00     Average reward: 23.54
Episode 20      Last reward: 89.00      Average reward: 31.96
Episode 30      Last reward: 108.00     Average reward: 76.43
Episode 40      Last reward: 20.00      Average reward: 83.57
Episode 50      Last reward: 19.00      Average reward: 56.94
Episode 60      Last reward: 53.00      Average reward: 48.44
Episode 70      Last reward: 147.00     Average reward: 82.04
Episode 80      Last reward: 90.00      Average reward: 94.94
Episode 90      Last reward: 144.00     Average reward: 97.71
Episode 100     Last reward: 200.00     Average reward: 133.73
Episode 110     Last reward: 200.00     Average reward: 158.69
Episode 120     Last reward: 159.00     Average reward: 162.60
Episode 130     Last reward: 150.00     Average reward: 159.57
Episode 140     Last reward: 195.00     Average reward: 163.27
Episode 150     Last reward: 143.00     Average reward: 157.88
Episode 160     Last reward: 113.00     Average reward: 151.82
Episode 170     Last reward: 147.00     Average reward: 146.14
Episode 180     Last reward: 199.00     Average reward: 150.11
Episode 190     Last reward: 200.00     Average reward: 168.77
Episode 200     Last reward: 200.00     Average reward: 177.60
Episode 210     Last reward: 102.00     Average reward: 174.29
Episode 220     Last reward: 189.00     Average reward: 171.91
Episode 230     Last reward: 200.00     Average reward: 169.92
Episode 240     Last reward: 200.00     Average reward: 181.99
Episode 250     Last reward: 200.00     Average reward: 189.22
Episode 260     Last reward: 200.00     Average reward: 188.75
Episode 270     Last reward: 180.00     Average reward: 192.27
Episode 280     Last reward: 200.00     Average reward: 175.83
Episode 290     Last reward: 200.00     Average reward: 185.53
Episode 300     Last reward: 200.00     Average reward: 191.33
Episode 310     Last reward: 200.00     Average reward: 194.81
Solved! Running reward is now 195.071295316 and the last episode runs to 200 time steps!
```

## 模型评估
强化学习模型一般采取边预测边训练，不断通过反馈来改进模型的学习方式，因此我们只需要观察学习过程中的reward变化情况，就可以评估强化学习算法的好坏。
在gym中，不同的游戏一般都设置了不同的solve_threshold，强化学习模型只需要达到这个threshold，即可完成训练。
