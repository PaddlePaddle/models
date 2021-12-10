# Linux GPU/CPU 服务化部署测试开发文档

# 目录

- [1. 简介](#1)
- [2. 命令与配置文件解析](#2)
    - [2.1 命令解析](#2.1)
    - [2.2 配置文件解析](#2.2)
- [3. 测试功能开发](#3)
    - [3.1 准备数据与环境](#3.1)
    - [3.2 准备开发所需脚本](#3.2)
    - [3.3 填写配置文件](#3.3)
    - [3.4 验证配置正确性](#3.4)
    - [3.5 撰写说明文档](#3.5)
- [4. 附录](#4)

<a name="1"></a>

## 1. 简介

本文档主要介绍飞桨模型在 Linux GPU/CPU 下服务化部署能力的测试开发过程。主要内容为

（1）参考 [《Linux GPU/CPU 基础训练推理开发文档》](../train_infer_python/README.md)，完成模型的训练和基于Paddle Inference的模型推理开发。

（2）参考[《Linux GPU/CPU 服务化部署功能开发文档》](./serving.md)，在Paddle Inference的模型推理基础上，完成服务化部署能力的开发。

（3）完成 TIPC 服务化部署测试开发（**本文档**）。


具体地，本文档主要关注Linux GPU/CPU 下模型的服务化部署能力，具体测试点如下：

- Inference 模型转 Serving 模型
- Paddle Inference 推理过程跑通

<a name="2"></a>

## 2. 命令与配置文件解析

<a name="2.1"></a>

### 2.1 命令解析

Serving部署过程中，通常需要用到下面3个命令。

```bash
# 模型转换：Inference 模型转为 Serving 模型
python3.7 -m paddle_serving_client.convert --dirname  ../../alexnet_infer/ --model_filename inference.pdmodel --params_filename inference.pdiparams --serving_server alexnet_server --serving_client alexnet_client

# 启动服务
python3.7 web_service.py

# 启动客户端访问
python3.7 pipeline_http_client.py --img-path=../../images/demo.jpg
```


在后续测试过程中，主要用到2个脚本文件和1个配置文件。

* `test_serving.sh`: 测试Serving服务化部署的脚本。**该脚本无需修改**。
* `model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt`: 配置文件模板，可以根据自己的Serving部署代码修改该配置文件，根据配置文件，会拼凑出希望执行的命令。

Serving服务化部署主要分为以下5个步骤。

<div align="center">
    <img src="./images/test_serving_pipeline.png" width="400">
</div>

其中设置了2个核验点。下面在第2章对配置文件进行详细说明，在第3章详细介绍开发过程。

<a name="2.2"></a>

### 2.2 配置文件解析

完整的`model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt`配置文件共有18行，包含3个方面的内容。

* 模型转化配置：第2~9行
* 服务端启动配置：第10~11行
* 客户端启动配置：第12~18行

具体内容见[model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt](./template/test/model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt)。

文本文件中主要有以下2种类型的字段。

* 一行内容以冒号为分隔符：该行可以被解析为`key:value`的格式，需要根据实际的含义修改该行内容，下面进行详细说明。
* 一行内容为`======xxxxx=====`：该行内容为注释信息，无需修改。

#### 2.2.1 模型转换配置参数

<details>
<summary><b>模型转换配置参数列表（点击以展开详细内容或者折叠）
</b></summary>

| 行号 | 参考内容                                             | 含义                  | key是否需要修改 | value是否需要修改 | 修改内容                        |
|----|---------------|---------------------|-----------|-------------|-----------------------------|
| 2  | model_name:your_model_name                       | 配置模型名字              | 否         | 是           | value修改为自己的模型名称             |
| 3  | python:python3.7                                 | 配置python命令          | 否         | 是           | value修改为自己的python命令         |
| 4  | trans_model:-m paddle_serving_client.convert     | 配置模型转化命令            | 否         | 否           | -                           |
| 5  | --dirname:./model_infer/                         | 配置Inference模型输入路径   | 否         | 是           | value修改为自己的Inference模型所在文件夹 |
| 6  | --model_filename:inference.pdmodel               | 配置模型结构文件名           | 否         | 是           | value修改为自己的模型结构文件名          |
| 7  | --params_filename:inference.pdiparams            | 配置模型参数文件名           | 否         | 是           | value修改为自己的模型参数文件名          |
| 8  | --serving_server:./deploy/serving/serving_server | 配置输出的Serving服务端模型路径 | 否         | 是           | value修改为自己的输出Serving服务端模型路径 |
| 9  | --serving_client:./deploy/serving/serving_client | 配置输出的Serving客户端模型路径 | 否         | 是           | value修改为自己的输出Serving客户端模型路径 |

</details>

以命令`python -m paddle_serving_client.convert --dirname  ./alexnet_infer/ --model_filename inference.pdmodel --params_filename inference.pdiparams --serving_server ./deploy/serving/alexnet_server --serving_client ./deploy/serving/alexnet_server`为例。

* Inference模型路径为`./alexnet_infer/`，因此第5行需要修改为`--dirname:./alexnet_infer/`
* 模型结构和参数文件名和默认保持一致，因此这里无需修改。
* 服务端保存的目录名称为`deploy/serving/alexnet_server`，因此第8行需要修改为`deploy/serving/alexnet_server`
* 客户端保存的目录名称为`deploy/serving/alexnet_client`，因此第9行需要修改为`deploy/serving/alexnet_client`


#### 2.2.2 服务启动配置参数

下面给出了配置文件中的训练命令配置参数（点击以展开详细内容或者折叠）

<details>
<summary><b>服务启动配置参数列表（点击以展开详细内容或者折叠）
</b></summary>

| 行号 | 参考内容                                                                      | 含义 | key是否需要修改 | value是否需要修改 | 修改内容                     |
|----|---------------------------------------------------------------------------|----|-----------|-------------|--------------------------|
| 10 | serving_dir:your_serving_dir                                              |    | 否         | 是           | value修改为自己的serving部署代码路径 |
| 11 | web_service:web_service.py |    | 否         | 否           | -                        |

</details>

在自动化测试时，会在设置的`serving_dir`目录下执行命令。假设Serving部署代码在目录`deploy/serving`下，运行命令为`python3.7 web_service.py`，则：

* Serving部署代码目录为`deploy/serving`，第10行需要修改为`serving_dir:deploy/serving`
* 服务启动脚本为`web_service.py`，和默认一致，第11行内容无需修改

#### 2.2.3 客户端访问启动配置参数

下面给出了配置文件中的客户端访问启动配置参数。

<details open>
<summary><b>客户端访问启动配置参数列表（点击以展开详细内容或者折叠）</b></summary>

| 行号 | 参考内容          | 含义       | key是否需要修改 | value是否需要修改 | 修改内容                        |
|----|-----------------|----------------|-----------|-------------|-----------------------------|
| 12 | op.model.local_service_conf.devices:0    | 配置GPU设备，后面的数字表示GPU ID | 否         | 否           | -                           |
| 13 | null:null | 配置是否使用mkldnn   | 否         | 否           | -                           |
| 14 | null:null        | 配置线程数          | 否         | 否           | -                           |
| 15 | null:null          | 配置是否使用TensorRT | 否         | 否           | -                           |
| 16 | null:null        | 配置预测精度         | 否         | 否           | -                           |
| 17 | pipline:pipeline_http_client.py                 | 客户端运行命令脚本      | 否         | 否           | -                           |
| 18 | --img_dir:../../images/demo.jpg                 | 配置图像目录         | 是         | 是           | key和value修改为自己的输入图像路径设置参数和值 |

</details>

以命令`python3.7 pipeline_http_client.py --img-path=../../images/demo.jpg`为例。

* 第12行配置GPU ID，默认使用0号卡，一般情况下无需修改。
* 第13~16行为服务端配置，无需修改。
* 第17行用于配置客户端命令，这里与默认相同，无需修改。
* 第18行配置传入的图片路径，上面命令中，图片路径是通过`--img-path=../../images/demo.jpg`传入，因此需要修改为`--img-path:../../images/demo.jpg`

<a name="3"></a>

## 3. 测试功能开发

<a name="3.1"></a>

### 3.1 准备小数据集与环境

**【基本内容】**

* 数据集：为方便快速验证服务化部署过程，需要准备至少1张图像用于测试，可以放在repo中。

* 环境：可以参考[Linux GPU/CPU 服务化部署功能开发规范](./serving.md)完成Serving部署环境的准备。


<a name="=3.2"></a>

### 3.2 准备开发所需脚本

**【基本内容】**

在repo中新建`test_tipc`目录，将文件 [test_serving.sh](template/test/template/test/test_serving.sh) 拷贝到`test_tipc`目录中。

**【注意事项】**

* 上述脚本文件无需改动，在实际使用时，直接修改配置文件即可。

<a name="3.3"></a>

### 3.3 填写配置文件

**【基本内容】**

在repo的`test_tipc/`目录中新建`configs/model_name`，将文件 [template/test/model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt](template/test/model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt) 拷贝到该目录中，其中`model_name`需要修改为您自己的模型名称。

**【实战】**

配置文件的含义解析可以参考 [2.2章节配置文件解析](#2.2) 部分。

AlexNet的测试开发配置文件可以参考：[model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/test_tipc/configs/AlexNet/model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt)。

<a name="3.4"></a>

### 3.4 验证配置正确性

**【基本内容】**

基于修改完的配置，运行

```bash
bash test_tipc/test_serving.sh ${your_params_file}
```

**【注意事项】**

如果运行失败，会输出具体的报错命令，可以根据输出的报错命令排查下配置文件的问题并修改。

**【实战】**

AlexNet中验证配置正确性的脚本：[AlexNet Serving 部署功能测试](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/test_tipc/docs/test_serving.md#23-%E6%B5%8B%E8%AF%95%E5%8A%9F%E8%83%BD)。


**【核验】**

基于修改后的配置文件，命令运行成功，测试通过。

<a name="3.5"></a>

### 3.5 撰写说明文档

**【基本内容】**

* 补充TIPC首页文档，补充更多部署方式。
* 撰写测试流程说明文档，说明文档模板为：[./template/test/test_serving.md](./template/test/test_serving.md)。

可以直接拷贝到自己的repo中，根据自己的模型进行修改。

**【实战】**

参考上面基本内容，具体地，以AlexNet为例，`test_tipc`文档如下所示。

1. TIPC功能总览文档：https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/test_tipc/README.md
2. Linux GPU/CPU 基础训练推理测试说明文档：https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/test_tipc/docs/test_serving.md

**【核验】**

repo中最终目录结构如下所示。

```
test_tipc
    |--configs                              # 配置目录
    |    |--model_name                      # 您的模型名称
    |           |--model_linux_gpu_normal_normal_serving_python_linux_gpu_cpu.txt   # Serving配置文件
    |--docs                                 # 文档目录
    |   |--test_serving.md                  # Serving测试说明文档
    |----README.md                          # TIPC 说明文档
    |----test_serving.sh                    # TIPC Serving测试解析脚本
    |----common_func.sh                     # TIPC 基础训练推理测试常用函数，无需改动
```

基于`test_serving.md`文档，跑通`Linux GPU/CPU Serving服务部署`流程。

<a name="4"></a>

## 4. 附录
