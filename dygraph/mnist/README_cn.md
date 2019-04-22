# MNIST
当我们学习编程的时候，编写的第一个程序一般是实现打印"Hello World"。而机器学习（或深度学习）的入门教程，一般都是 MNIST 数据库上的手写识别问题。原因是手写识别属于典型的图像分类问题，比较简单，同时MNIST数据集也很完备。
本页将介绍如何使用PaddlePaddle在DyGraph模式下实现MNIST，包括[安装](#installation)、[训练](#training-a-model)、[输出](#log)。

---
## 内容
- [安装](#installation)
- [训练](#training-a-model)
- [输出](#log)

## 安装

在当前目录下运行样例代码需要PadddlePaddle Fluid的v1.4.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据安装文档中的说明来更新PaddlePaddle。

## 训练
教程中使用`paddle.dataset.mnist`数据集作为训练数据，可以通过如下的方式启动训练：
```
env CUDA_VISIBLE_DEVICES=0 python mnist_dygraph.py
```

## 输出
执行训练开始后，将得到类似如下的输出。
```
batch_id 0,loss 2.1786134243
batch_id 10,loss 0.898496925831
batch_id 20,loss 1.32524681091
...
```
