# Linux GPU/CPU 基础训练推理测试开发规范

## 1. 本规范测试功能

本规范主要关注Linux GPU/CPU 下模型的基础训练推理全流程测试，具体测试点如下：

- 模型训练方面：单机单卡训练跑通
- 飞桨模型转换：保存静态图模型
- Paddle Inference 推理过程跑通

目的：对于特定模型，通过一个配置文件，跑通该模型的训练、模型转换和Paddle Inference推理过程。

## 2. 前置条件

### 2.1 准备小数据集与环境

**【基本内容】**

为方便快速验证训练/评估/推理过程，需要准备一个小数据集（训练集和验证集各8~16张图像即可，压缩后数据大小建议在`20M`以内），放在`lite_data`文件夹下。

相关文档可以参考[论文复现赛指南3.2章节](../lwfx/ArticleReproduction_CV.md)，代码可以参考`基于ImageNet准备小数据集的脚本`：[prepare.py](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step2/prepare.py)。

**【注意事项】**

* 为方便管理，建议在上传至github前，首先将lite_data文件夹压缩为tar包，直接上传tar包即可，在测试训练评估与推理过程时，可以首先对数据进行解压。
    * 压缩命令： `tar -zcf lite_data.tar lite_data`
    * 解压命令： `tar -xf lite_data.tar`

### 2.2 规范训练日志

**【基本内容】**

训练日志符合规范，相关文档可以参考[论文复现赛指南3.12章节](../lwfx/ArticleReproduction_CV.md)。


**【注意事项】**

* 训练日志中，至少需要包含`epoch`, `iter`, `loss`, `avg_reader_cost`, `avg_batch_cost`, `avg_ips`关键字段及信息。
* 规范化日志输出建议使用[AutoLog](https://github.com/LDOUBLEV/AutoLog)工具包，示例代码可以参考：[AlexNet训练过程中添加训练日志](https://github.com/littletomatodonkey/AlexNet-Prod/blob/0099e0bb74020ef8e8c11e5811517c47bf1e78ef/pipeline/Step5/AlexNet_paddle/train.py#L204)。

### 2.3 支持模型推理功能，规范推理日志

**【基本内容】**

模型支持动转静与推理过程，推理日志符合日志规范。

**【注意事项】**

* 相关内容可以参考[模型推理开发规范](./inference.md)。

## 3. 配置文件解析与开发

### 3.1 介绍

运行模型训练、动转静、推理过程的命令差别很大，但是都可以拆解为3个部分：

```
python   run_script    set_configs
```

例如：

* 对于通过配置文件传参的场景来说，`python3.7 train.py -c config.yaml -o epoch_num=120`
    * `python`部分为`python3.7`
    * `run_script`部分为`train.p`
    * `set_configs`部分为`-c config.yaml -o epoch_num=120`
* 对于通过argparse传参的场景来说，`python3.7 train.py --lr=0.1, --output=./output`
    * `python`部分为`python3.7`
    * `run_script`部分为`train.p`
    * `set_configs`部分为`--lr=0.1, --output=./output`

给定配置文件模板，便可以拼凑为完整的训练命令格式。


### 3.2 配置文件解析与开发

完整的配置文件共有51行，包含5个方面的内容。

* 训练参数：第1~14行
* 训练脚本配置：第15~22行
* 评估脚本和配置：第23~26行（本部分无需关注，这里不再展开介绍）
* 模型导出脚本和配置：第27~36行
* 模型Inference推理：第37~51行

具体内容见[train_infer_python_params.txt](params_template/train_infer_python_params.txt)。

文本文件中主要有以下3种类型字段。

* 一行内容以冒号为分隔符：该行可以被解析为`key-value`的格式，需要根据实际的含义修改该行内容，下面进行详细说明。
* 一行内容为`======xxxxx=====`：该行内容表示，无需修改。
* 一行内容为`##`：该行内容表示段落分隔符，没有实际意义，无需修改。

#### 3.2.1 训练配置参数

在配置文件中，可以通过下面的方式配置一些常用的输出文件，如：是否使用GPU、迭代轮数、batch size、预训练模型路径等，下面给出了常用的配置以及需要修改的内容。


| 行号 | 参考内容                                | 含义            | key是否需要修改 | value是否需要修改 | 修改内容                             |
|----|-------------------------------------|---------------|-----------|-------------|----------------------------------|
| 2  | model_name:alexnet                  | 模型名字          | 否         | 是           | value修改为自己的模型名字                  |
| 3  | python:python3.7                    | python环境      | 否         | 是           | value修改为自己的python环境              |
| 4  | gpu_list:0                          | gpu id        | 否         | 是           | value修改为自己的GPU ID                |
| 5  | use_gpu:True                        | 是否使用GPU       | 是         | 是           | key修改为设置GPU使用的内容，value修改         |
| 6  | auto_cast:null                      | 是否使用混合精度      | 否         | 否           | -                                |
| 7  | epoch_num:lite_train_infer=1        | 迭代的epoch数目    | 是         | 否           | key修改为可以设置代码中epoch数量的内容          |
| 8  | output_dir:./output/                | 输出目录          | 是         | 否           | key修改为代码中可以设置输出路径的内容             |
| 9  | train_batch_size:lite_train_infer=1 | 训练的batch size | 是         | 否           | key修改为可以设置代码中batch size的内容       |
| 10 | pretrained_model:null               | 预训练模型         | 是         | 是           | 如果训练时指定了预训练模型，则需要key和value需要对应修改 |
| 11 | train_model_name:latest             | 训练结果的模型名字     | 否         | 是           | value需要修改为训练完成之后保存的模型名称，用于后续的动转静 |
| 12 | null:null                           | 预留字段          | 否         | 否           | -                                |
| 13 | null:null                           | 预留字段          | 否         | 否           | -                                |


#### 3.2.2 训练命令配置

| 行号 | 参考内容                                        | 含义              | key是否需要修改 | value是否需要修改 |                   |
|----|---------------------------------------------|-----------------|-----------|-------------|-------------------|
| 15 | trainer:norm_train                          | 训练方法            | 否         | 否           | -                 |
| 16 | norm_train:tools/train.py -c config.yaml -o | norm_train的训练脚本 | 否         | 是           | value可以修改为自己的训练命令 |
| 17 | pact_train:null                             | 量化训练脚本配置        |           | 否           | -                 |
| 18 | fpgm_train:null                             | 裁剪训练脚本配置        | 否         | 否           | -                 |
| 19 | distill_train:null                          | 蒸馏训练脚本配置        | 否         | 否           | -                 |
| 20 | null:null                                   | 预留字段            | 否         | 否           | -                 |
| 21 | null:null                                   | 预留字段            | 否         | 否           | -                 |


#### 3.2.3 模型导出配置参数


| 行号 | 参考内容                                | 含义            | key是否需要修改 | value是否需要修改 | 修改内容                             |
|----|-------------------------------------|---------------|-----------|-------------|----------------------------------|
| 2  | model_name:alexnet                  | 模型名字          | 否         | 是           | value修改为自己的模型名字                  |
| 3  | python:python3.7                    | python环境      | 否         | 是           | value修改为自己的python环境              |
| 4  | gpu_list:0                          | gpu id        | 否         | 是           | value修改为自己的GPU ID                |
| 5  | use_gpu:True                        | 是否使用GPU       | 是         | 是           | key修改为设置GPU使用的内容，value修改         |
| 6  | auto_cast:null                      | 是否使用混合精度      | 否         | 否           | -                                |
| 7  | epoch_num:lite_train_infer=1        | 迭代的epoch数目    | 是         | 否           | key修改为可以设置代码中epoch数量的内容          |
| 8  | output_dir:./output/                | 输出目录          | 是         | 否           | key修改为代码中可以设置输出路径的内容             |
| 9  | train_batch_size:lite_train_infer=1 | 训练的batch size | 是         | 否           | key修改为可以设置代码中batch size的内容       |
| 10 | pretrained_model:null               | 预训练模型         | 是         | 是           | 如果训练时指定了预训练模型，则需要key和value需要对应修改 |
| 11 | train_model_name:latest             | 训练结果的模型名字     | 否         | 是           | value需要修改为训练完成之后保存的模型名称，用于后续的动转静 |
| 12 | null:null                           | 预留字段          | 否         | 否           | -                                |
| 13 | null:null                           | 预留字段          | 否         | 否           | -                                |


#### 3.2.4 模型推理配置


| 行号 | 参考内容                                                 | 含义              | key是否需要修改 | value是否需要修改 | 修改内容                              |
|----|------------------------------------------------------|-----------------|-----------|-------------|-----------------------------------|
| 37 | infer_model:null                                     | 推理模型保存路径        | 否         | 否           |                                   |
| 38 | infer_export:tools/export_model.py -c config.yaml -o | 推理前是否需要导出       | 否         | 是           | value修改为和30行内容一致即可                |
| 39 | infer_quant:False                                    | 是否量化推理          | 否         | 否           | -                                 |
| 40 | inference:infer.py                                   | 推理脚本            | 否         | 是           | value修改为自己的推理脚本                   |
| 41 | --use_gpu:True|False                                 | 是否使用GPU         | 是         | 是           | key和value修改为GPU设置的参数和值            |
| 42 | --use_mkldnn:True|False                              | 是否使用MKLDNN      | 是         | 是           | key和value修改为MKLNN设置的参数和值          |
| 43 | --cpu_threads:1|6                                    | 开启MKLDNN时使用的线程数 | 是         | 是           | key和value修改为CPU线程数设置的参数和值         |
| 44 | --batch_size:1                                       | 推理batch size    | 是         | 否           | key修改为代码中可以修改为batch size的内容       |
| 45 | --use_tensorrt:null                                  | 是否开启TensorRT预测  | 否         | 否           | 不对该项进行测试                          |
| 46 | --precision:null                                     | 精度范围            | 否         | 否           | 不对该项进行测试                          |
| 47 | --model_dir:./infer                                  | 模型目录            | 是         | 否           | key修改为代码中可以设置inference model目录的内容 |
| 48 | --image_dir:./lite_data/1.jpg                        | 图片路径或者图片文件夹路径   | 是         | 否           | key和value修改为自己的CPU线程数设置参数和值       |
| 49 | --save_log_path:null                                 | 推理日志输出路径        | 否         | 否           | -                                 |
| 50 | --benchmark:False                                    | 是否使用            | 是         | 是           | key和value修改为规范化推理日志输出设置的参数和值      |
| 51 | null:null                                            | 预留字段            | 否         | 否           | -                                 |


### 3.3 验收

#### 3.3.1 目录结构

整理目录结构如下所示。可以参考：[AlexNet_paddle/test_tipc](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/test_tipc)


```
test_tipc
    |--configs                              # 配置目录
    |    |--model_name                      # 您的模型名称
    |           |--train_infer_python.txt   # 基础训练推理测试配置文件
    |--docs                                 # 文档目录
    |   |--test_train_inference_python.md   # 基础训练推理测试说明文档
    |----README.md                          # TIPC说明文档
    |----test_train_inference_python.sh     # TIPC基础训练推理测试解析脚本，无需改动
    |----common_func.sh                     # TIPC基础训练推理测试常用函数，无需改动
```

#### 3.3.2 配置文件

* 修改配置参数模板中文件内容，制作您的配置参数文件，放在`configs`目录下。
* 可以参考AlexNet的[train_infer_python.txt](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/test_tipc/configs/AlexNet/train_infer_python.txt)。

#### 3.3.3 说明文档

* `test_tipc/docs/test_train_inference_python.md`文档中需要包含环境(`AutoLog安装`)与数据准备(如果需要，如数据解压等)、一键测试的命令。按照文档流程可以走通训练推理过程。

* TIPC首页文档可以参考：[README.md](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/test_tipc/README.md)，需要将其中的模型与支持的训练推理模式修改为该模型目前测试脚本中包含的模式。

* TIPC训练推理测试说明文档可以参考：[test_train_inference_python.md](https://github.com/littletomatodonkey/AlexNet-Prod/blob/tipc/pipeline/Step5/AlexNet_paddle/test_tipc/docs/test_train_inference_python.md)，需要将其中的模型与支持的训练推理模式修改为该模型目前测试脚本中包含的模式。



## 4. 附录

### 4.1 common_func.sh 函数介绍

common_func.sh 中包含了5个函数，会被test_train_inference_python.sh 等执行脚本调用，分别是：
```
func_parser_key() 解析params.txt中：前的部分

func_parser_value() 解析params.txt中：后的部分

func_set_params()  会反馈key=value的字符串，用与组建参数命令，如果key或者value为null，则返回空字符，即这个参数不会生效

func_parser_params() 为了解析和测试模式MODE相关的参数，目前只用来解析epoch和batch size参数

status_check() 状态检查函数，获取上条指令运行的状态，如果是0，则运行成功，如果是其他则运行失败，失败和成功的指令都会存放在results.log文件中
```
同一个模型要测试的组合可以通过一个txt文件管理，如下，构建了一个test_tipc/configs/ppocr_det_mobile/train_infer_python.txt文件 用于管理训练配置。

![](./images/train_infer_params_train.png)
上述部分是test_train_inference_python.sh运行所需要的参数部分，包含正常训练脚本（第16行），量化训练执行脚本（17行），训练所需配置的参数（第4-10行）

在执行test_train_inference_python.sh脚本执行的时候，会解析对应行的参数，如下：
```
dataline=$(awk 'NR==1, NR==51{print}'  $FILENAME)
```
首先，FILENAME变量传入的是参数文件，从参数文件中获取到params.txt中的所有参数，如果是lite_train_lite_infer模式下，获取FILENAME参数文件第1行到51行的参数。

参数解析格式通过func_parser_params和func_parser_value函数完成，如下：
```
# The training params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
gpu_list=$(func_parser_value "${lines[3]}")
train_use_gpu_key=$(func_parser_key "${lines[4]}")
train_use_gpu_value=$(func_parser_value "${lines[4]}")
autocast_list=$(func_parser_value "${lines[5]}")
autocast_key=$(func_parser_key "${lines[5]}")
epoch_key=$(func_parser_key "${lines[6]}")
epoch_num=$(func_parser_params "${lines[6]}")
save_model_key=$(func_parser_key "${lines[7]}")
train_batch_key=$(func_parser_key "${lines[8]}")
train_batch_value=$(func_parser_params "${lines[8]}")
pretrain_model_key=$(func_parser_key "${lines[9]}")
pretrain_model_value=$(func_parser_value "${lines[9]}")
train_model_name=$(func_parser_value "${lines[10]}")
train_infer_img_dir=$(func_parser_value "${lines[11]}")
train_param_key1=$(func_parser_key "${lines[12]}")
train_param_value1=$(func_parser_value "${lines[12]}")
```

解析到的训练参数在test_train_inference_python.sh中的第263-269行，通过`func_set_params`完成参数配置，`func_set_params`有两个输入分别是key value，最终返回的格式为key=value，比如：
```
# case1
cpp_batch_size_key="batch_size"
batch_size="10"
set_batchsize=$(func_set_params "${cpp_batch_size_key}" "${batch_size}")

echo $set_batchsize
# batch_size=10 # 输出结果是字符串batch_size=10

# case2
cpp_batch_size_key="batch_size"
batch_size="null"
set_batchsize=$(func_set_params "${cpp_batch_size_key}" "${batch_size}")
echo $set_batchsize
#  # 输出结果是空字符
```
如果value为空格，或者null，则`func_set_params`返回的结果为空格，在组建命令的时候，这个参数就不会生效，因此，如果不希望设置某个参数，可以把参数设置为null。

最终，所有参数可以组合出一个完整的命令，组合方式：

![](./images/train_infer_params_train_run.png)

上图中273行即是组合出的命令，python对应python3.7,  run_train  可以是`ppocr_det_mobile_params.txt`中的`norm_train`，`quant_train`参数后的执行脚本，即要执行正常训练的脚本还是执行量化训练的脚本等等。
