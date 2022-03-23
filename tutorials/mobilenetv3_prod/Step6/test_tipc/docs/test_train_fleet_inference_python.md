# Linux GPU/CPU 多机多卡训练推理测试

Linux GPU/CPU 多机多卡训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 多机多卡 |
|  :----: |   :----:  |    :----:  |
|  MobileNetV3  | mobilenet_v3_small | 分布式训练 |


- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  MobileNetV3   |  mobilenet_v3_small |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备环境
- 准备至少两台可以相互`ping`通的机器

  这里推荐使用Docker容器的方式来运行。以Paddle2.2.2 GPU版，cuda10.2 cudnn7为例：
  ```
  拉取预安装 PaddlePaddle 的镜像：
  nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7

  用镜像构建并进入Docker容器：
  nvidia-docker run --name paddle -it --net=host -v $PWD:/paddle registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7 /bin/bash
  ```
  不同的物理机环境配置，安装请参照[官网安装说明](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)。

- 安装依赖
    ```
    pip install  -r requirements.txt
    ```

- 安装AutoLog（规范化日志输出工具）
    ```
    pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ```

### 2.2 功能测试

首先修改配置文件中的`ip`设置: 假设两台机器的`ip`地址分别为`192.168.0.1`和`192.168.0.2`，则对应的配置文件`gpu_list`字段需要修改为`gpu_list:192.168.0.1,192.168.0.2;0,1`。`ip`地址查看命令为`ifconfig`。

测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_lite_infer
```
**注意：** 多机多卡的训练推理测试有别于单机，需要在各个节点上分别启动命令。

以`mobilenet_v3_small`的`Linux GPU/CPU 多机多卡训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/prepare.sh test_tipc/configs/mobilenet_v3_small/train_fleet_infer_python.txt lite_train_lite_infer
```

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/mobilenet_v3_small/train_fleet_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```bash
Run successfully with command - python3.7 -m paddle.distributed.launch --ips=192.168.0.1,192.168.0.2 --gpus=0,1 train.py --output-dir=./log/mobilenet_v3_small/lite_train_lite_infer/norm_train_gpus_0,1_nodes_2 --epochs=5   --batch-size=4!
......
Run successfully with command - python3.7 deploy/inference_python/infer.py --use-gpu=False --model-dir=./log/mobilenet_v3_small/lite_train_lite_infer/norm_train_gpus_0,1_nodes_2 --batch-size=1   --benchmark=True > ./log/mobilenet_v3_small/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1 !
```

在开启benchmark参数时，可以得到测试的详细数据，包含运行环境信息（系统版本、CUDA版本、CUDNN版本、驱动版本），Paddle版本信息，参数设置信息（运行设备、线程数、是否开启内存优化等），模型信息（模型名称、精度），数据信息（batchsize、是否为动态shape等），性能信息（CPU,GPU的占用、运行耗时、预处理耗时、推理耗时、后处理耗时），内容如下所示：

```
[2022/03/22 06:15:51] root INFO: ---------------------- Env info ----------------------
[2022/03/22 06:15:51] root INFO:  OS_version: Ubuntu 16.04
[2022/03/22 06:15:51] root INFO:  CUDA_version: 10.2.89
[2022/03/22 06:15:51] root INFO:  CUDNN_version: 7.6.5
[2022/03/22 06:15:51] root INFO:  drivier_version: 440.64.00
[2022/03/22 06:15:51] root INFO: ---------------------- Paddle info ----------------------
[2022/03/22 06:15:51] root INFO:  paddle_version: 2.2.2
[2022/03/22 06:15:51] root INFO:  paddle_commit: b031c389938bfa15e15bb20494c76f86289d77b0
[2022/03/22 06:15:51] root INFO:  log_api_version: 1.0
[2022/03/22 06:15:51] root INFO: ----------------------- Conf info -----------------------
[2022/03/22 06:15:51] root INFO:  runtime_device: cpu
[2022/03/22 06:15:51] root INFO:  ir_optim: True
[2022/03/22 06:15:51] root INFO:  enable_memory_optim: True
[2022/03/22 06:15:51] root INFO:  enable_tensorrt: False
[2022/03/22 06:15:51] root INFO:  enable_mkldnn: False
[2022/03/22 06:15:51] root INFO:  cpu_math_library_num_threads: 1
[2022/03/22 06:15:51] root INFO: ----------------------- Model info ----------------------
[2022/03/22 06:15:51] root INFO:  model_name: classification
[2022/03/22 06:15:51] root INFO:  precision: fp32
[2022/03/22 06:15:51] root INFO: ----------------------- Data info -----------------------
[2022/03/22 06:15:51] root INFO:  batch_size: 1
[2022/03/22 06:15:51] root INFO:  input_shape: dynamic
[2022/03/22 06:15:51] root INFO:  data_num: 1
[2022/03/22 06:15:51] root INFO: ----------------------- Perf info -----------------------
[2022/03/22 06:15:51] root INFO:  cpu_rss(MB): 227.2812, gpu_rss(MB): None, gpu_util: None%
[2022/03/22 06:15:51] root INFO:  total time spent(s): 0.1583
[2022/03/22 06:15:51] root INFO:  preprocess_time(ms): 18.6493, inference_time(ms): 139.591, postprocess_time(ms): 0.0875
```

该信息可以在运行log中查看，以`mobilenet_v3_small`为例，log位置在`./log/mobilenet_v3_small/lite_train_lite_infer/python_infer_gpu_batchsize_1.log`。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
