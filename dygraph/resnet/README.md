DyGraph模式下Residual Network实现
========

简介
--------
Residual Network（ResNet）是常用的图像分类模型。我们实现了在paddlepaddle的DyGraph模式下相应的实现。可以对比原先静态图下实现（[Residual Network](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)）来了解paddle中DyGraph模式。
运行本目录下的程序示例需要使用PaddlePaddle develop最新版本。如果您的PaddlePaddle安装版本低于此要求，请按照[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)中的说明更新PaddlePaddle安装版本。

动态图文档请见[Dygraph](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/user_guides/howto/dygraph/DyGraph.html)


## 代码结构
```
└── train.py     # 训练脚本。
```

## 使用的数据

教程中使用`paddle.dataset.flowers`数据集作为训练数据，该数据集通过`paddle.dataset`模块自动下载到本地。

## 训练测试Residual Network

在GPU单卡上训练Residual Network:

```
env CUDA_VISIBLE_DEVICES=0 python train.py
```

这里`CUDA_VISIBLE_DEVICES=0`表示是执行在0号设备卡上，请根据自身情况修改这个参数。

Paddle动态图支持多进程多卡进行模型训练，启动训练的方式：
```
python -m paddle.distributed.launch --selected_gpus=0,1,2,3  --log_dir ./mylog train.py   --use_data_parallel 1
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
执行训练开始后，将得到类似如下的输出。每一轮`batch`训练将会打印当前epoch、step以及loss值。当前默认执行`epoch=10`, `batch_size=8`。您可以调整参数以得到更好的训练效果，同时也意味着消耗更多的内存（显存）以及需要花费更长的时间。
```text
epoch id: 0, batch step: 0, loss: 4.951202
epoch id: 0, batch step: 1, loss: 5.268410
epoch id: 0, batch step: 2, loss: 5.123999
```
