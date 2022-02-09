# Linux GPU/CPU 基础训练推理测试开发文档

# 目录

- [1. 简介](#1)
- [2. 基本训练推理功能测试开发](#2)
    - [2.1 准备待测试的命令](#2.1)
    - [2.2 准备数据与环境](#2.2)
    - [2.3 准备开发所需脚本](#2.3)
    - [2.4 填写配置文件](#2.4)
    - [2.5 验证配置正确性](#2.5)
    - [2.6 撰写说明文档](#2.6)
- [3. FAQ](#3)

<a name="1"></a>

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了飞桨训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）在Linux GPU/CPU 基础训练推理测试下的开发流程。

<a name="2"></a>

## 2. 基础训练推理功能测试开发

Linux GPU/CPU 下的基础训练推理测试开发的过程可以分为6个步骤，如下图所示。

<div align="center">
    <img src="./images/test_linux_train_infer_python_pipeline.png" width="400">
</div>

其中设置了2个核验点，分别为

* 验证配置正确性
* 撰写说明文档

<a name="2.1"></a>

### 2.1 准备待测试的命令

Linux端基础训练推理功能测试的主程序为`tutorials/mobilenetv3_prod/Step6/test_tipc/test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

首先需要进入到目录`tutorials/mobilenetv3_prod/Step6/`下，测试命令如下，如果希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

**【准备数据】**

```bash
bash test_tipc/prepare.sh ${your_params_file} lite_train_lite_infer
```

**【测试】**

```bash
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_lite_infer
```

以`mobilenet_v3_small`的`Linux GPU/CPU 基础训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/prepare.sh test_tipc/configs/mobilenet_v3_small/train_infer_python.txt lite_train_lite_infer
```

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/mobilenet_v3_small/train_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```
Run successfully with command - xxx
```

<a name="2.2"></a>

### 2.2 准备数据与环境

**【数据】**

用于分类模型基础训练推理测试的数据位于`tutorials/mobilenetv3_prod/Step6/test_images/lite_data.tar`，直接解压即可。

**【环境】**

- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    pip install paddlepaddle-gpu==2.2.0
    # 安装CPU版本的Paddle
    pip install paddlepaddle==2.2.0
    ```

- 安装依赖
    ```
    cd tutorials/mobilenetv3_prod/Step6/
    pip3 install  -r requirements.txt
    ```

- 安装AutoLog（规范化日志输出工具）
    ```
    git clone https://github.com/LDOUBLEV/AutoLog
    cd AutoLog
    pip install -r requirements.txt
    python setup.py bdist_wheel
    pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
    cd ../
    ```

<a name="2.3"></a>

### 2.3 准备开发所需脚本

开发脚本所在目录是：`tutorials/mobilenetv3_prod/Step6/test_tipc/`
主要涉及以下几个文件：
`common_func.sh`: TIPC基础训练推理测试常用函数，无需改动
`prepare.sh `   : TIPC基础训练推理测试数据准备脚本，如果需要测试其他数据或需要额外的预训练模型，可以按照模板进行准备数据和所需的预训练模型
`test_train_inference_python.sh`: TIPC基础训练推理测试解析脚本，无需改动

<a name="2.4"></a>

### 2.4 填写配置文件

如果需要新增测试模型，需要在 `tutorials/mobilenetv3_prod/Step6/test_tipc/configs` 目录下新建一个以模型名字命名的目录`model_name`，然后在该模型目录下新建一个`train_infer_python.txt`，该文件是基础训练推理测试的配置文件。
模板文件参考`tutorials/mobilenetv3_prod/Step6/test_tipc/configs/mobilenet_v3_small/train_infer_python.txt`，可以基于该文件来修改对应的参数使其适配自己的模型。


<a name="2.5"></a>

### 2.5 验证配置正确性

根据以上步骤进行配置，可以得到以下目录结构，该结构位于`tutorials/mobilenetv3_prod/Step6/`目录下

```
test_tipc
    |--configs                              # 配置目录
    |    |--model_name                      # 您的模型名称
    |           |--train_infer_python.txt   # 基础训练推理测试配置文件
    |--docs                                 # 文档目录
    |   |--test_train_inference_python.md   # 基础训练推理测试说明文档
    |----README.md                          # TIPC说明文档
    |----prepare.sh                         # TIPC基础训练推理测试数据准备脚本
    |----test_train_inference_python.sh     # TIPC基础训练推理测试解析脚本，无需改动
    |----common_func.sh                     # TIPC基础训练推理测试常用函数，无需改动
```

根据测试文档，基于配置文件，跑通训练推理全流程测试。

<a name="2.6"></a>

### 2.6 撰写说明文档

验证配置正确后，需要补充一下说明文档。主要涉及以下两个文档：

- `test_tipc/README.md` 文档中对该模型支持的的功能进行总体介绍。
- `test_tipc/docs/test_train_inference_python.md` 文档中对Linux GPU/CPU 基础训练推理的功能支持情况进行介绍。

<a name="3"></a>

## 3. FAQ