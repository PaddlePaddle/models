# Linux GPU/CPU C++ 推理功能测试开发文档

# 目录

- [1. 简介](#1)
- [2. 命令与配置文件解析](#2)
    - [2.1 命令解析](#2.1)
    - [2.2 配置文件和运行命令映射解析](#2.2)
- [3. 基本C++推理功能测试开发](#3)
    - [3.1 准备系统环境](#3.1)
    - [3.2 准备输入数据和推理模型](#3.2)
    - [3.3 准备推理所需代码](#3.3)
    - [3.4 编译得到可执行代码](#3.4)
    - [3.5 运行得到结果](#3.5)
    - [3.6 填写配置文件](#3.6)
    - [3.7 验证配置正确性](#3.7)
    - [3.8 撰写说明文档](#3.8)
- [4. FAQ](#4)

<a name="1"></a>

## 1. 简介

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT进行预测加速，从而实现更优的推理性能。
更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)。
本文档主要介绍飞桨模型在 Linux GPU/CPU 下基于C++预测引擎的推理过程开发。

<a name="2"></a>

## 2. 命令与配置文件解析

<a name="2.1"></a>

### 2.1 命令解析

基于paddle inference的C++预测命令如下：
```
run_scripts configs_path img_path
```

* `run_scripts`：最终编译好的可执行命令。
* `configs_path`：设置模型路径、是否使用GPU、是否开启mkldnn、是否开启TensorRT等。
* `img_path`：待预测的图像路径。

<a name="2.2"></a>

### 2.2 配置文件解析
完整的`inference_cpp.txt`配置文件共有14行，包含两个方面的内容。
* 运行环境参数配置：第1~8行
* 模型参数配置：第10~14行

具体内容见[inference_cpp.txt](../../mobilenetv3_prod/Step6/test_tipc/configs/mobilenet_v3_small/inference_cpp.txt)

配置文件中主要有以下2种类型的字段。

* 一行内容以空格为分隔符：该行可以被解析为`key value`的格式，需要根据实际的含义修改该行内容，下面进行详细说明。
* 一行内容为`# xxxxx`：该行内容为注释信息，无需修改。

<details>
<summary><b>配置参数（点击以展开详细内容或者折叠）
</b></summary>

| 行号 | 参考内容                                | 含义            | key是否需要修改 | value是否需要修改 | 修改内容                             |
|----|-------------------------------------|---------------|-----------|-------------|----------------------------------|
| 2  | use_gpu      | 是否使用GPU    | 否         | 是           | value根据是否使用GPU进行修改               |
| 3  | gpu_id       | 使用的GPU卡号  | 否         | 是           | value修改为自己的GPU ID              |
| 4  | gpu_mem      | 显存          | 否         | 是           | value修改为自己的GPU 显存             |
| 5  | cpu_math_library_num_thread | 底层科学计算库所用线程的数量  | 否      | 是           | value修改为合适的线程数         |
| 6  | use_mkldnn   | 是否使用MKLDNN加速    | 否        | 是          | value根据是否使用MKLDNN进行修改          |
| 7  | use_tensorrt | 是否使用tensorRT进行加速          | 否         | 是           | value根据是否使用tensorRT进行修改             |
| 8  | use_fp16 | 是否使用半精度浮点数进行计算，该选项仅在use_tensorrt为true时有效 | 否         | 是          | value根据在开启tensorRT时是否使用半精度进行修改|
| 11 | cls_model_path  | 预测模型结构文件路径         | 否         | 是           | value修改为预测模型结构文件路径 |
| 12 | cls_params_path | 预测模型参数文件路径  | 否         | 是           | vvalue修改为预测模型参数文件路径 |
| 13 | resize_short_size  | 预处理时图像缩放大小         | 否         | 是           | value修改为预处理时图像缩放大小  
| 14 | crop_size          | 预处理时图像裁剪后的大小      | 否         | 是           | value修改为预处理时图像裁剪后的大小  


</details>

<a name="3"></a>

## 3. 基本C++推理功能测试开发

基于Paddle Inference的推理过程可以分为5个步骤，如下图所示。
<div align="center">
    <img src="../images/infer_cpp.png" width="600">
</div>
其中设置了2个核验点，分别为

* 准备输入数据和推理模型
* 编译得到可执行代码

<a name="3.1"></a>

### 3.1 准备系统环境

该部分可参考 [文档](../../mobilenetv3_prod/Step6/test_tipc/docs/test_inference_cpp.md) 中的2.2.1，2.2.1，2.2.3章节准备环境。

<a name="3.2"></a>

### 3.2 准备输入数据和推理模型
该部分可参考 [文档](../../mobilenetv3_prod/Step6/test_tipc/docs/test_inference_cpp.md) 中的2.1章节准备数据和推理模型。

<a name="3.3"></a>

### 3.3 准备推理所需代码
基于预测引擎的推理过程包含4个步骤：初始化预测引擎、预处理、推理、后处理。参考[文档](./infer_cpp.md)准备预测引擎推理代码并编译成功。

<a name="3.5"></a>

### 3.5 运行得到结果

相关脚本位置[run.sh](../../mobilenetv3_prod/Step6/deploy/inference_cpp/tools/run.sh)
```bash
./build/clas_system ./tools/config.txt ../../images/demo.jpg
```

<a name="3.6"></a>

### 3.6 填写配置文件
**【基本内容】**

在repo的`test_tipc/`目录中新建`configs/model_name`，将文件 [inference_cpp.txt](../../mobilenetv3_prod/Step6/test_tipc/configs/mobilenet_v3_small/inference_cpp.txt) 拷贝到该目录中，其中`model_name`需要修改为您自己的模型名称。

**【实战】**

配置文件的含义解析可以参考 [2.2节配置文件解析](#2.2) 部分。

mobilenet_v3_small的测试开发配置文件可以参考：[inference_cpp.txt](../../mobilenetv3_prod/Step6/test_tipc/configs/mobilenet_v3_small/inference_cpp.txt)。

<a name="3.7"></a>

### 3.7 验证配置正确性

**【基本内容】**

基于修改完的配置，运行

```bash
bash test_tipc/test_inference_cpp.sh ${your_params_file}
```

**【注意事项】**

如果运行失败，会输出具体的报错命令，可以根据输出的报错命令排查下配置文件的问题并修改，示例报错如下所示。

```
Run failed with command - ./deploy/inference_cpp/build/clas_system test_tipc/configs/mobilenet_v3_small/inference_cpp.txt ./images/demo.jpg > ./log/infer_cpp/infer_cpp_use_cpu_use_mkldnn.log 2>&1 !
```

**【实战】**

以mobilenet_v3_small的`Linux GPU/CPU C++推理功能测试` 为例，命令如下所示。

```bash
bash test_tipc/test_inference_cpp.sh test_tipc/configs/mobilenet_v3_small/inference_cpp.txt
```

输出结果如下，表示命令运行成功。

```bash
Run successfully with command - ./deploy/inference_cpp/build/clas_system test_tipc/configs/mobilenet_v3_small/inference_cpp.txt ./images/demo.jpg > ./log/infer_cpp/infer_cpp_use_cpu_use_mkldnn.log 2>&1 !
```

也可以在`./log/infer_cpp/infer_cpp_use_cpu_use_mkldnn.log`中查看详细的输出结果。

**【核验】**

基于修改后的配置文件，测试通过，全部命令成功

<a name="3.8"></a>

### 3.8 撰写说明文档

**【基本内容】**

撰写TIPC功能总览和测试流程说明文档，分别为

1. TIPC功能总览文档：test_tipc/README.md
2. Linux GPU/CPU C++推理功能测试说明文档：test_tipc/docs/test_inference_cpp.md

2个文档模板分别位于下述位置，可以直接拷贝到自己的repo中，根据自己的模型进行修改。

1. [README.md](../../mobilenetv3_prod/Step6/test_tipc/README.md)
2. [test_inference_cpp](../../mobilenetv3_prod/Step6/test_tipc/docs/test_inference_cpp.md)

**【实战】**

mobilenet_v3_small中`test_tipc`文档如下所示。

1. TIPC功能总览文档：[README.md](../../mobilenetv3_prod/Step6/test_tipc/README.md)
2. Paddle2ONNX 测试说明文档：[test_inference_cpp.md](../../mobilenetv3_prod/Step6/test_tipc/docs/test_inference_cpp.md)

**【核验】**

repo中最终目录结构如下所示。

```
test_tipc
    |--configs                              # 配置目录
    |    |--model_name                      # 您的模型名称
    |           |--inference_cpp.txt        # inference_cpp测试配置文件
    |--docs                                 # 文档目录
    |   |--test_inference_cpp.md            # inference_cpp测试说明文档
    |----README.md                          # TIPC说明文档
    |----test_inference_cpp.sh              # TIPC inference_cpp解析脚本，无需改动
    |----common_func.sh                     # TIPC基础训练推理测试常用函数，无需改动
```

基于`test_inference_cpp.md`文档，跑通`inference_cpp功能测试`流程。

<a name="4"></a>

## 4. FAQ
