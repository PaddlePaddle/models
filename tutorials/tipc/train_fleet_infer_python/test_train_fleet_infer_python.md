# Linux GPU/CPU 多机多卡训练推理测试开发文档

# 目录
- [1. 简介](#1)
- [2. 命令与配置文件解析](#2)
- [3. 多机多卡训练推理测试开发](#3)
- [4. FAQ](#4)

<a name="1"></a>

## 1. 简介

本文档主要关注Linux GPU/CPU 下模型的多机多卡训练推理全流程功能测试。与基础训练推理测试类似，其具体测试点如下：

- 模型训练：多机多卡训练跑通
- 模型动转静：保存静态图模型跑通
- 模型推理：推理过程跑通

<a name="2"></a>

## 2. 命令与配置文件解析

此章节可以参考[基础训练推理测试开发文档](../train_infer_python/test_train_infer_python.md#2)。 **主要的差异点**为脚本的第4行、第13行和第14行，如下所示：
| 行号 | 参考内容                                        | 含义              | key是否需要修改 | value是否需要修改 |  修改内容                 |
|----|---------------------------------------------|-----------------|-----------|-------------|-------------------|
| 4  | gpu_list:xx.xx.xx.xx,yy.yy.yy.yy;0,1     | 节点IP地址和GPU ID        | 否         | 是           | value修改为自己的IP地址和GPU ID                |
| 13 | trainer:fleet_train                          | 训练方法            | 否         | 否           | -                 |
| 14 | fleet_train:train.py                         | 多机多卡训练脚本 | 否         | 是           | value可以修改为自己的训练命令 |

以训练命令`python3.7 -m paddle.distributed.launch --ips 192.168.0.1,192.168.0.2 --gpus 0,1 train.py --device=gpu --epochs=1 --data-path=./lite_data`为例，该命令为多机多卡训练（非裁剪、量化、蒸馏等方式），运行在`ip`地址为`192.168.0.1`和`192.168.0.2`的`0,1`号卡上，
因此:
* 配置文件的第4行写`gpu_list:192.168.0.1,192.168.0.2;0,1`。
* 配置文件的第13行`trainer`内容为`fleet_train`, 区别于基础训练的`normal_train`、混合精度训练的`amp_train`。
* 配置文件的第14行内容为`fleet_train:train.py`。

<a name="3"></a>

## 3. 多机多卡训练推理功能测试开发

多机多卡训练推理功能测试开发过程，同样包含了如下6个步骤。

<div align="center">
    <img src="../train_infer_python/images/test_linux_train_infer_python_pipeline.png" width="600">
</div>

其中设置了2个核验点，详细的开发过程与[基础训练推理测试开发](../train_infer_python/test_train_infer_python.md#3)类似。**主要的差异点**有如下四处:

* ### 1）准备验证环境

该步骤需要准备至少两台可以相互`ping`通的机器。这里推荐使用Docker容器的方式来运行。以Paddle2.2.2 GPU版，cuda10.2, cudnn7为例：
```
拉取预安装 PaddlePaddle 的镜像：
nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7

用镜像构建并进入Docker容器：
nvidia-docker run --name paddle -it --net=host -v $PWD:/paddle registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7 /bin/bash
```
不同的物理机环境配置，Docker容器创建请参照[官网安装说明](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html#old-version-anchor-2-%E5%AE%89%E8%A3%85%E6%AD%A5%E9%AA%A4)。

* ### 2）增加配置文件

此处需要将文件 [train_fleet_infer_python.txt](../../mobilenetv3_prod/Step6/test_tipc/configs/mobilenet_v3_small/train_fleet_infer_python.txt) 拷贝到`test_tipc/configs/model_name`路径下，`model_name`为您自己的模型名字。同时，需要相应
修改`train_fleet_infer_python.txt`模板文件中的`model_name`字段。

* ### 3）验证配置正确性

首先，修改配置文件中的`ip`设置:  假设两台机器的`ip`地址分别为`192.168.0.1`和`192.168.0.2`，则对应的配置文件`gpu_list`字段需要修改为`gpu_list:192.168.0.1,192.168.0.2;0,1`，`ip`地址查看命令为`ifconfig`。

基于修改完的配置，运行

```bash
bash test_tipc/prepare.sh ${your_params_file} lite_train_lite_infer
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_lite_infer
```
**注意:** 多机多卡的训练推理验证过程有别于单机，需要在各个节点上分别启动命令。

以mobilenet_v3_small的`Linux GPU/CPU 多机多卡训练推理功能测试` 为例，命令如下所示。

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/mobilenet_v3_small/train_fleet_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。
```bash
Run successfully with command - python3.7 -m paddle.distributed.launch --ips=192.168.0.1,192.168.0.2 --gpus=0,1 train.py --output-dir=./log/mobilenet_v3_small/lite_train_lite_infer/norm_train_gpus_0,1_nodes_2 --epochs=5   --batch-size=4!
......
Run successfully with command - python3.7 deploy/inference_python/infer.py --use-gpu=False --model-dir=./log/mobilenet_v3_small/lite_train_lite_infer/norm_train_gpus_0,1_nodes_2 --batch-size=1   --benchmark=True > ./log/mobilenet_v3_small/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1 !
```
若基于修改后的配置文件，全部命令都运行成功，则验证通过。

* ### 4）撰写说明文档

此处需要增加`Linux GPU/CPU 多机多卡训练推理功能测试`说明文档，该文档的模板位于[test_train_fleet_inference_python.md](../../mobilenetv3_prod/Step6/test_tipc/docs/test_train_fleet_inference_python.md)，可以直接拷贝到自己的repo中，根据自己的模型进行修改。

若已完成多机多卡训练测试开发以及基础训练测试的开发，则repo最终目录结构如下所示。
```
test_tipc
    |--configs                                  # 配置目录
    |    |--model_name                          # 您的模型名称
    |           |--train_infer_python.txt       # 基础训练推理测试配置文件
    |           |--train_fleet_infer_python.txt # 多机多卡训练推理测试配置文件
    |--docs                                     # 文档目录
    |   |--test_train_inference_python.md       # 基础训练推理测试说明文档
    |   |--test_train_fleet_inference_python.md # 多机多卡训练推理测试说明文档
    |----README.md                              # TIPC说明文档
    |----prepare.sh                             # TIPC基础、多机多卡训练推理测试数据准备脚本
    |----test_train_inference_python.sh         # TIPC基础、多机多卡训练推理测试解析脚本
    |----common_func.sh                         # TIPC基础、多机多卡训练推理测试常用函数
```
最后，自行基于`test_train_fleet_inference_python.md`文档，跑通`Linux GPU/CPU 多机多卡训练推理功能测试`流程即可。

<a name="4"></a>

## 4. FAQ
