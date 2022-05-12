# Linux GPU/CPU PYTHON 服务化部署测试开发文档

# 目录

- [1. 简介](#1)
- [2. 命令与配置文件解析](#2)
    - [2.1 命令解析](#2.1)
    - [2.2 配置文件和运行命令映射解析](#2.2)
- [3. 基本训练推理功能测试开发](#3)
    - [2.1 准备待测试的命令](#3.1)
    - [2.2 准备数据与环境](#3.2)
    - [2.3 准备开发所需脚本](#3.3)
    - [2.4 填写配置文件](#3.4)
    - [2.5 验证配置正确性](#3.5)
    - [2.6 撰写说明文档](#3.6)
- [4. FAQ](#4)

<a name="1"></a>

## 1. 简介

本文档主要关注Linux GPU/CPU 下模型的PYTHON 服务化部署功能测试，具体测试点如下：

- 模型转换：部署模型转换跑通
- 模型部署：python服务部署过程跑通

为了一键跑通上述所有功能，本文档提供了`训推一体全流程`功能自动化测试工具，它包含3个脚本文件和1个配置文件，分别是：

* `test_serving_infer_python.sh`: 测试部署模型转换和python服务部署预测的脚本，会对`serving_infer_python.txt`进行解析，得到具体的执行命令。**该脚本无需修改**。
* `prepare.sh`: 准备测试需要的数据或需要的预训练模型。
* `common_func.sh`: 在配置文件一些通用的函数，如配置文件的解析函数等，**该脚本无需修改**。
* `serving_infer_python.txt`: 配置文件，其中的内容会被`test_serving_infer_python.sh`解析成具体的执行命令字段。

<a name="2"></a>

## 2. 命令与配置文件解析

<a name="2.1"></a>

### 2.1 命令解析

部署模型转换和服务部署的运行命令差别很大，但是都可以拆解为3个部分：

```
python  run_script   set_configs
```

例如：

* 对于通过argparse传参的场景来说，`python3.7 pipeline_http_client.py --image_dir=../../lite_data/test/`
    * `python`部分为`python3.7`
    * `run_script`部分为`pipeline_http_client.py`
    * `set_configs`部分为`--image_dir=../../lite_data/test/`

其中，可修改参数`set_configs`一般通过`=`进行分隔，`=`前面的内容可以认为是key，后面的内容可以认为是value，那么通过给定配置文件模板，解析配置，得到其中的key和value，结合`python`和`run_script`，便可以组合出一条完整的命令。

<a name="2.2"></a>

### 2.2 配置文件和运行命令映射解析

完整的`serving_infer_python.txt`配置文件共有13行，包含2个方面的内容。

* Serving 部署模型转换：第4~10行
* Serving 启动部署服务：第10~13行

具体内容见[serving_infer_python.txt](../../mobilenetv3_prod/Step6/test_tipc/configs/mobilenet_v3_small/serving_infer_python.txt)

配置文件中主要有以下3种类型的字段。

* 一行内容以冒号为分隔符：该行可以被解析为`key:value`的格式，需要根据实际的含义修改该行内容，下面进行详细说明。
* 一行内容为`======xxxxx=====`：该行内容为注释信息，无需修改。
* 一行内容为`##`：该行内容表示段落分隔符，没有实际意义，无需修改。

#### 2.2.1 模型转换配置参数

在配置文件中，可以通过下面的方式配置一些常用的超参数，如：Paddle模型路径、部署模型路径等，下面给出了常用的训练配置以及需要修改的内容。

<details>
<summary><b>模型转换配置参数（点击以展开详细内容或者折叠）
</b></summary>


| 行号 | 参考内容                                | 含义            | key是否需要修改 | value是否需要修改 | 修改内容                             |
|----|-------------------------------------|---------------|-----------|-------------|----------------------------------|
| 2  | model_name:mobilenet_v3_small      | 模型名字          | 否         | 是           | value修改为自己的模型名字                  |
| 3  | python:python3.7                    | python环境      | 否         | 是           | value修改为自己的python环境              |
| 5  | --dirname:./inference/mobilenet_v3_small_infer/ | Paddle inference 模型保存路径 | 否        | 是           | value修改为自己 Inference 模型的路径                  |
| 6  | --model_filename:inference.pdmodel  | pdmodel 文件名     | 否     | 是           | value修改为 pdmodel 文件名           |
| 7  | --params_filename:inference.pdiparams     | pdiparams 文件名 | 否        | 是       | value修改为 pdiparams 文件名       |
| 8  | --serving_server:./deploy/serving_infer_python.serving_server/ | 转换出的部署模型目录 | 否       | 是           | value修改为部署模型模型保存路径      |
| 9 | --serving_client:./deploy/serving_infer_python.serving_client/ | 转换出的服务模型目录 | 否       | 是           | value修改为服务模型保存路径      |
</details>

以模型转换命令 `python3.7 -m paddle_serving_client.convert --dirname=./inference/resnet50_infer/ --model_filename=inference.pdmodel --params_filename=inference.pdiparams --serving_server=./deploy/serving_infer_python.serving_server/ --serving_client=./deploy/serving_infer_python.serving_client/` 为例，总共包含5个超参数。

* inference 模型路径： `--dirname=./inference/resnet50_infer/` 则需要修改第5行， 修改后内容为`--dirname:./inference/resnet50_infer/`。
* pdmodel文件名： `--model_filename=inference.pdmodel ` 则需要修改第6行， 修改后内容为 `--model_filename:inference.pdmodel`。
* 其他参数以此类推


#### 2.2.2 python服务部署配置参数

下面给出了配置文件中的python服务部署配置参数（点击以展开详细内容或者折叠）

<details>
<summary><b>服务部署配置参数（点击以展开详细内容或者折叠）
</b></summary>

| 行号 | 参考内容                                | 含义            | key是否需要修改 | value是否需要修改 | 修改内容                             |
|----|-------------------------------------|---------------|-----------|-------------|----------------------------------|
| 10  | serving_dir:./deploy/serving_infer_python.| python部署执行目录    | 否     | 是           | value修改为python部署工作目录 |
| 11  | web_service:web_service.py | 启动部署服务命令     | 否     | 是           | value修改为自定义的服务部署脚本 |
| 12 | pipline:pipeline_http_client.py | 启动访问客户端    | 否     | 是           | value修改为自定义的客户端启动脚本  |
| 13 | --image_dir:../../lite_data/test/ | 预测图片路径    | 否     | 是           | value修改为预测图片路径           |


</details>

以启动客户端命令 `python3.7 pipeline_http_client.py --image_dir=./my_data/test_img.png` 为例，总共包含1个超参数。

* 预测图片路径：`--image_dir=./my_data/test_img.png`， 则需要修改配置文件的第13行，`key`为`--image_dir`， `value` 为 `./my_data/test_img.png`，修改后内容为`--image_dir=:./my_data/test_img.png`。


## 3. python 服务化部署功能测试开发

服务化部署功能测试开发主要分为以下6个步骤。

<div align="center">
    <img src="https://raw.githubusercontent.com/PaddlePaddle/models/release/2.2/tutorials/tipc/train_infer_python/images/test_linux_train_infer_python_pipeline.png" width="800">
</div>


其中设置了2个核验点，下面详细介绍开发过程。

<a name="3.1"></a>

### 3.1 准备待测试的命令

**【基本内容】**

准备模型转换、模型推理的命令，后续会将这些命令按照[第2节](#2)所述内容，映射到配置文件中。

**【实战】**

MobileNetV3的Serving模型转换、服务部署运行命令如下所示。

```bash
# 模型转换
python3.7 -m paddle_serving_client.convert
--dirnam=./inference/mobilenet_v3_small_infer/ \
--model_filename=inference.pdmodel \
--params_filename=inference.pdiparams \
--serving_server=./deploy/serving_infer_python.serving_server/ \
--serving_client=./deploy/serving_infer_python.serving_client/
# 部署
python3.7 web_service.py
python3.7 pipeline_http_client.py --image_dir=../../lite_data/test/
```

<a name="3.2"></a>

### 3.2 准备数据与环境

**【基本内容】**

1. 数据集：为方便快速验证训练/评估/推理过程，需要准备一个小数据集（训练集和验证集各8~16张图像即可，压缩后数据大小建议在`20M`以内），放在`lite_data`文件夹下。

    相关文档可以参考[论文复现赛指南3.2章节](../../../docs/lwfx/ArticleReproduction_CV.md)，代码可以参考`基于ImageNet准备小数据集的脚本`：[prepare.py](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step2/prepare.py)。

2. 环境：安装好PaddlePaddle即可进行离线量化训练推理测试开发

**【注意事项】**

* 为方便管理，建议在上传至github前，首先将lite_data文件夹压缩为tar包，直接上传tar包即可，在测试训练评估与推理过程时，可以首先对数据进行解压。
    * 压缩命令： `tar -zcf lite_data.tar lite_data`
    * 解压命令： `tar -xf lite_data.tar`


<a name="3.3"></a>

### 3.3 准备开发所需脚本

**【基本内容】**

在repo中新建`test_tipc`目录，将文件 [common_func.sh](../../mobilenetv3_prod/Step6/test_tipc/common_func.sh) ， [prepare.sh](../../mobilenetv3_prod/Step6/test_tipc/prepare.sh) 和 [test_serving_infer_python.sh](../../mobilenetv3_prod/Step6/test_tipc/test_serving_infer_python.sh) 分别拷贝到`test_tipc`目录中。


**【注意事项】**

* 上述3个脚本文件无需改动，在实际使用时，直接修改配置文件即可。

<a name="3.4"></a>

### 3.4 填写配置文件

**【基本内容】**

在repo的`test_tipc/`目录中新建`configs/model_name`，将文件 [serving_infer_python.txt](../../mobilenetv3_prod/Step6/test_tipc/configs/mobilenet_v3_small/serving_infer_python.txt) 拷贝到该目录中，其中`model_name`需要修改为您自己的模型名称。

**【实战】**

配置文件的含义解析可以参考 [2.2节配置文件解析](#2.2) 部分。

mobilenet_v3_small的测试开发配置文件可以参考：[serving_infer_python.txt](../../mobilenetv3_prod/Step6/test_tipc/configs/mobilenet_v3_small/serving_infer_python.txt)。

<a name="3.5"></a>

### 3.5 验证配置正确性

**【基本内容】**

基于修改完的配置，运行

```bash
bash test_tipc/prepare.sh ${your_params_file} serving_infer
bash test_tipc/test_serving_infer_python.sh ${your_params_file} serving_infer
```

**【注意事项】**

如果运行失败，会输出具体的报错命令，可以根据输出的报错命令排查下配置文件的问题并修改，示例报错如下所示。

```
Run failed with command - python3.7 pipeline_http_client.py > ../../log/mobilenet_v3_small/serving_infer/server_infer_gpu_batchsize_1.log 2>&1 !
```

**【实战】**

以mobilenet_v3_small的`Linux GPU/CPU 离线量化训练推理功能测试` 为例，命令如下所示。

```bash
bash test_tipc/test_serving_infer_python.sh test_tipc/configs/mobilenet_v3_small/serving_infer_python.txt serving_infer
```

输出结果如下，表示命令运行成功。

```bash
Run successfully with command - python3.7 pipeline_http_client.py > ../../log/mobilenet_v3_small/serving_infer/server_infer_gpu_batchsize_1.log 2>&1 !
```

**【核验】**

基于修改后的配置文件，测试通过，全部命令成功

<a name="3.6"></a>

### 3.6 撰写说明文档

**【基本内容】**

撰写TIPC功能总览和测试流程说明文档，分别为

1. TIPC功能总览文档：test_tipc/README.md
2. Linux GPU/CPU 离线量化训练推理功能测试说明文档：test_tipc/docs/test_serving_infer_python.md

2个文档模板分别位于下述位置，可以直接拷贝到自己的repo中，根据自己的模型进行修改。

1. [README.md](../../mobilenetv3_prod/Step6/test_tipc/README.md)
2. [test_serving_infer_python.md](../../mobilenetv3_prod/Step6/test_tipc/docs/test_serving_infer_python.md)

**【实战】**

mobilenet_v3_small中`test_tipc`文档如下所示。

1. TIPC功能总览文档：[README.md](../../mobilenetv3_prod/Step6/test_tipc/README.md)
2. Python Serving 测试说明文档：[test_serving_infer_python.md](../../mobilenetv3_prod/Step6/test_tipc/docs/test_serving_infer_python.md)

**【核验】**

repo中最终目录结构如下所示。

```
test_tipc
    |--configs                              # 配置目录
    |    |--model_name                      # 您的模型名称
    |           |--serving_infer_python.txt   # python服务化部署测试配置文件
    |--docs                                 # 文档目录
    |   |--test_serving_infer_python.md   # python服务化部署测试说明文档
    |----README.md                          # TIPC说明文档
    |----test_serving_infer_python.sh     # TIPC python服务化部署解析脚本，无需改动
    |----common_func.sh                     # TIPC基础训练推理测试常用函数，无需改动
```

基于`test_serving_infer_python.md`文档，跑通`python服务化部署功能测试`流程。

<a name="4"></a>

## 4. FAQ
