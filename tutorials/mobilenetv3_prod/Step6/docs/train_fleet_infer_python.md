# MobileNetV3

## 目录

- [1. 简介](#1)
- [2. 分布式训练](#2)
  - [2.1 单机多卡训练](#2.1)
  - [2.2 多机多卡训练](#2.2)
- [3. FAQ](#3)

<a name="1"></a>

## 1. 简介

飞桨分布式从产业实践出发，提供包括数据并行、模型并行和流水线并行等在内的完备的并行能力，提供简单易用地分布式训练接口和丰富的底层通信原语，赋能用户业务发展。

本文，我们以最常用的数据并行为例，介绍Linux GPU多机多卡从训练到推理的使用。

<a name="2"></a>

## 2. 分布式训练

<a name="2.1"></a>



### 2.1 单机多卡训练

当使用单机多卡时，可以通过如下的命令启动分布式训练任务：

```shell
python -m paddle.distributed.launch --gpus="0,1" --log_dir=./log train.py
```

其中，``--gpus``选项指定用户分布式训练使用的GPU卡，``--log_dir``参数指定用户日志的保存目录。

<a name="2.2"></a>

### 2.2 多机多卡训练

1. 我们以2台机器为例，说明如何启动多机多卡分布式训练任务。假设两台机器的ip地址分别为192.168.0.1和192.168.0.2。
   
   首先，我们需要确保两台机器间的网络是互通的，可以通过``ping``命令验证机器间网络的互通性，如下所示：
   
   ```shell
   # 在ip地址为192.168.0.1的机器上
   ping 192.168.0.2
   ```
   
   接着，我们分别在两台机器上启动分布式任务：
   
   ```shell
   # 在ip地址为192.168.0.1的机器上
   python -m paddle.distributed.launch --ips="192.168.0.1,192.168.0.2" --gpus="0,1" train.py
   ```
   
   ```shell
   # 在ip地址为192.168.0.2的机器上
   python -m paddle.distributed.launch --ips="192.168.0.1,192.168.0.2" --gpus="0,1" train.py
   ```
   
   启动上述命令后，将在控制台上输出类似如下所示的信息：
   
   ```shell
   WARNING 2021-01-04 17:59:08,725 launch.py:314] Not found distinct arguments and compiled with cuda. Default use collective mode
   launch train in GPU mode
   INFO 2021-01-04 17:59:08,727 launch_utils.py:472] Local start 4 processes. First process distributed environment info (Only For Debug):
       +=======================================================================================+
       |                        Distributed Envs                      Value                    |
       +---------------------------------------------------------------------------------------+
       |                 PADDLE_CURRENT_ENDPOINT                 192.168.0.1:17901               |
       |                     PADDLE_TRAINERS_NUM                        2                      |
       |                PADDLE_TRAINER_ENDPOINTS         192.168.0.1:17901,192.168.0.0.1:18846...       |
       |                     FLAGS_selected_gpus                        0                      |
       |                       PADDLE_TRAINER_ID                        0                      |
       +=======================================================================================+
   
   ...
   W0104 17:59:19.018365 43338 device_context.cc:342] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 9.2
   W0104 17:59:19.022523 43338 device_context.cc:352] device: 0, cuDNN Version: 7.4.
   W0104 17:59:23.193490 43338 fuse_all_reduce_op_pass.cc:78] Find all_reduce operators: 161. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 
   ```

当使用paddle.distributed.launch模块启动分布式任务时，所有日志将保存在./log目录下，日志文件名为workerlog.xx，其中xx为整数；每个卡训练进程对应一个日志文件。

<a name="3"></a>

## 3. FAQ
