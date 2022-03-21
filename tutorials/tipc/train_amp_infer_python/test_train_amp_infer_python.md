# Linux GPU/CPU 混合精度训练推理测试开发文档

# 目录

- [1. 简介](#1)
- [2. 命令与配置文件解析](#2)
- [3. 混合精度训练推理功能测试开发](#3)
- [4. FAQ](#4)

<a name="1"></a>

## 1. 简介

本文档主要关注Linux GPU/CPU 下模型的混合精度训练推理全流程功能测试。与基础训练推理测试类似，其具体测试点如下：

- 模型训练：单机单卡/多卡训练跑通
- 模型动转静：保存静态图模型
- 模型推理：推理过程跑通

为了一键跑通上述所有功能，需要提供`训推一体全流程`功能自动化测试工具，它包含3个脚本文件和1个配置文件，分别是：

* `test_train_inference_python.sh`: 测试Linux上训练、模型动转静、推理功能的脚本，会对`train_amp_infer_python.txt`进行解析，得到具体的执行命令。
* `prepare.sh`: 准备测试需要的数据或需要的预训练模型。
* `common_func.sh`: 在配置文件一些通用的函数，如配置文件的解析函数等。
* `train_amp_infer_python.txt`: 配置文件，其中的内容会被`test_train_inference_python.sh`解析成具体的执行命令字段。

**注意**: 通常情况下，我们是先完成基础训练推理的开发和测试，再集成混合精度训练功能及相应的测试开发。若您已完成基础训练推理功能的开发和测试，则只需要添加混合精度训练推理相关的配置文件即可。`test_train_inference_python.sh`，`prepare.sh`，`common_func.sh`这3个脚本文件无需修改。

<a name="2"></a>

## 2. 命令与配置文件解析

此章节可以参考[基础训练推理测试开发文档](../train_infer_python/test_train_infer_python.md#2)。 **主要的差异点**为脚本的第13行和第14行，如下所示：
| 行号 | 参考内容                                        | 含义              | key是否需要修改 | value是否需要修改 |  修改内容                 |
|----|---------------------------------------------|-----------------|-----------|-------------|-------------------|
| 13 | trainer:amp_train                          | 训练方法            | 否         | 否           | -                 |
| 14 | amp_train:train.py --amp_level=O1          | 混合精度训练脚本 | 否         | 是           | value可以修改为自己的训练命令 |

以训练命令`python3.7 train.py --amp_level=O1 --device=gpu --epochs=1 --data-path=./lite_data`为例，该命令为混合精度训练（非裁剪、量化、蒸馏等方式），因此

* 配置文件的第13行写`amp_train`, 区别于基础训练的`normal_train`。
* 配置文件的第14行内容为`amp_train:train.py --amp_level=O1`，区别于基础训练的`normal_train:train.py`。

**注意**，模板配置文件中默认测试混合精度训练的`O1`模式，若您需要测试`O2`模式，只需要将配置文件第14行的`amp_train:train.py --amp_level=O1`改为`amp_train:train.py --amp_level=O2`即可。 `O1`模式和`O2`模式的区别详见官网文档[自动混合精度训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/amp_cn.html#sanshiyongfeijiangkuangjiashixianzidonghunhejingdu)

<a name="3"></a>

## 3. 混合精度训练推理功能测试开发

混合精度训练推理功能测试开发过程，同样包含了如下6个步骤。

<div align="center">
    <img src="./images/test_linux_train_amp_infer_python_pipeline.png" width="800">
</div>

其中设置了2个核验点，详细的开发过程与[基础训练推理测试开发](../train_infer_python/test_train_infer_python.md#3)类似。**主要的差异点**有如下三处:

* ### 1） 增加配置文件

此处需要将文件 [train_amp_infer_python.txt](../../mobilenetv3_prod/Step6/test_tipc/configs/mobilenet_v3_small/train_amp_infer_python.txt) 拷贝到`test_tipc/configs/model_name`路径下，`model_name`为您自己的模型名字。同时，需要相应
修改`train_amp_infer_python.txt`模板文件中的`model_name`字段。

* ### 2）验证配置正确性

基于修改完的配置，运行

```bash
bash test_tipc/prepare.sh ${your_params_file} lite_train_lite_infer
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_lite_infer
```

以mobilenet_v3_small的`Linux GPU/CPU 混合精度训练推理功能测试` 为例，命令如下所示。

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/mobilenet_v3_small/train_amp_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```bash
Run successfully with command - python3.7 --amp_level=O1 train.py --output-dir=./log/mobilenet_v3_small/lite_train_lite_infer/norm_train_gpus_0 --epochs=5   --batch-size=4!
......
Run successfully with command - python3.7 deploy/inference_python/infer.py --use-gpu=False --model-dir=./log/mobilenet_v3_small/lite_train_lite_infer/norm_train_gpus_0,1 --batch-size=1   --benchmark=False > ./log/mobilenet_v3_small/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1 !
```

若基于修改后的配置文件，全部命令都运行成功，则验证通过。


* ### 3）撰写说明文档

此处需要增加`Linux GPU/CPU 混合精度训练推理功能测试`说明文档，该文档的模板位于[test_train_amp_inference_python.md](../../mobilenetv3_prod/Step6/test_tipc/docs/test_train_amp_inference_python.md)，可以直接拷贝到自己的repo中，根据自己的模型进行修改。


若已完成混合精度训练测试开发以及基础训练测试的开发，则repo最终目录结构如下所示。
```
test_tipc
    |--configs                                  # 配置目录
    |    |--model_name                          # 您的模型名称
    |           |--train_infer_python.txt       # 基础训练推理测试配置文件
    |           |--train_amp_infer_python.txt   # 混合精度训练推理测试配置文件
    |--docs                                     # 文档目录
    |   |--test_train_inference_python.md       # 基础训练推理测试说明文档
    |   |--test_train_amp_inference_python.md   # 混合精度训练推理测试说明文档
    |----README.md                              # TIPC说明文档
    |----prepare.sh                             # TIPC基础、混合精度训练推理测试数据准备脚本
    |----test_train_inference_python.sh         # TIPC基础、混合精度训练推理测试解析脚本
    |----common_func.sh                         # TIPC基础、混合精度训练推理测试常用函数
```
最后，自行基于`test_train_amp_inference_python.md`文档，跑通`Linux GPU/CPU 混合精度训练推理功能测试`流程即可。

<a name="4"></a>

## 4. FAQ
