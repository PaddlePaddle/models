SE_ResNeXt
===========

简介
--------
SE (Sequeeze-and-Excitation) block 并不是一个完整的网络结构，而是一个子结构，可以嵌到其他分类或检测模型中。SENet block 和 ResNeXt 的结合在 ILSVRC 2017 的分类项目中取得 了第一名的成绩。在 ImageNet 数据集上将 top-5 错误率从原先的最好成绩 2.991% 降低到 2.251%。

运行本目录下的程序示例需要使用PaddlePaddle develop最新版本。如果您的PaddlePaddle安装版本低于此要求，请按照[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)中的说明更新PaddlePaddle安装版本。

动态图文档请见[Dygraph](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/user_guides/howto/dygraph/DyGraph.html)


## 代码结构
```
└── train.py     # 训练脚本。
```

## 使用的数据

教程中使用`paddle.dataset.flowers`数据集作为训练数据，该数据集通过`paddle.dataset`模块自动下载到本地。

## 训练测试Residual Network

在GPU单卡上训练 Network:

```
env CUDA_VISIBLE_DEVICES=0 python train.py
```

这里`CUDA_VISIBLE_DEVICES=0`表示是执行在0号设备卡上，请根据自身情况修改这个参数。

亦可以使用多卡进行训练：
```
python -m paddle.distributed.launch --selected_gpus=0,1,2,3 --log_dir ./mylog train.py   --use_data_parallel 1
```
这里`--selected_gpus=0,1,2,3`表示使用0，1，2，3号设备卡，共计4卡进行多卡训练，请根据自身情况修改这个参数。
此时，程序会将每个进程的输出log导入到`./mylog`路径下：
```
.
├── mylog
│   ├── workerlog.0
│   ├── workerlog.1
│   ├── workerlog.2
│   └── workerlog.3
├── README.md
└── train.py
```


## 输出
执行训练开始后，将得到类似如下的输出。每一轮`batch`训练将会打印当前epoch、step以及loss值。当前默认执行`epoch=10`, `batch_size=64`。您可以调整参数以得到更好的训练效果，同时也意味着消耗更多的内存（显存）以及需要花费更长的时间。

```text
epoch 0 | batch step 0, loss 4.594 acc1 0.000 acc5 0.078 lr 0.01250
epoch 0 | batch step 10, loss 4.499 acc1 0.067 acc5 0.153 lr 0.01250
epoch 0 | batch step 20, loss 4.536 acc1 0.051 acc5 0.139 lr 0.01250
epoch 0 | batch step 30, loss 4.532 acc1 0.048 acc5 0.141 lr 0.01250
```
