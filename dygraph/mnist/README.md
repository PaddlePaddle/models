# MNIST
当我们学习编程的时候，编写的第一个程序一般是实现打印"Hello World"。而机器学习（或深度学习）的入门教程，一般都是 MNIST 数据库上的手写识别问题。原因是手写识别属于典型的图像分类问题，比较简单，同时MNIST数据集也很完备。
本页将介绍如何使用PaddlePaddle在DyGraph模式下实现MNIST，包括[安装](#installation)、[训练](#training-a-model)、[输出](#log)、[参数保存](#save)、[模型评估](#evaluation)。

动态图文档请见[Dygraph](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/user_guides/howto/dygraph/DyGraph.html)

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
env CUDA_VISIBLE_DEVICES=0 python train.py
```
Paddle动态图支持多进程多卡进行模型训练，启动训练的方式：
```
python -m paddle.distributed.launch --selected_gpus=0,1,2,3 --log_dir ./mylog train.py   --use_data_parallel 1
```
此时，程序会将每个进程的输出log导入到`./mylog`路径下：
```
.
├── mylog
│   ├── workerlog.0
│   ├── workerlog.1
│   ├── workerlog.2
│   └── workerlog.3
├── README.md
└── train.py
```

## 输出
执行训练开始后，将得到类似如下的输出。
```
Loss at epoch 0 step 0: [2.3043773]
Loss at epoch 0 step 100: [0.20764539]
Loss at epoch 0 step 200: [0.18648806]
Loss at epoch 0 step 300: [0.10279777]
Loss at epoch 0 step 400: [0.03940877]
...
```

## 参数保存
调用`fluid.dygraph.save_persistables()`接口可以把模型的参数进行保存。
```python
fluid.dygraph.save_persistables(mnist.state_dict(), "save_dir")
```

## 测试
执行`mnist.eval()`可以切换至评估状态，即不更新只使用参数进行训练，通过这种方式进行测试或者评估。
```python
mnist.eval()
```

## 模型评估
我们使用手写数据集中的一张图片来进行评估。为了区别训练模型，我们使用`with fluid.dygraph.guard()`来切换到一个新的参数空间，然后构建一个用于评估的网络`mnist_infer`，并通过`mnist_infer.load_dict()`来加载使用`fluid.dygraph.load_persistables`读取的参数。然后用`mnist_infer.eval()`切换到评估。
```python
with fluid.dygraph.guard():

    mnist_infer = MNIST("mnist")
    # load checkpoint
    mnist_infer.load_dict(
        fluid.dygraph.load_persistables("save_dir"))

    # start evaluate mode
    mnist_infer.eval()
```
如果无意外，将可以看到预测的结果：
```text
Inference result of image/infer_3.png is: 3
```
