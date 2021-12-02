# Paddle2ONNX 测试开发规范

- [1. 总览](#1)
  * [1.1 背景](#11)
  * [1.2 训推一体自动化测试](#12)
  * [1.3 文本检测样板间概览](#13)
- [2. 训推一体规范接入 Paddle2ONNX 预测流程](#2)
  * [2.1 准备数据和环境](#21)
  * [2.2 规范化输出预测日志](#22)
    + [2.2.1 预测日志规范](#221)
    + [2.2.2 接入步骤](#222)
  * [2.3 编写自动化测试代码](#23)
- [3. 附录](#3)
  * [3.1 自动化测试脚本test_paddle2onnx.sh 函数介绍](#31)
  * [3.2 其他说明](#32)

<a name="1"></a>
# 1、总览

<a name="11"></a>
## 1.1 背景
训推一体 CI 机制，旨在监控框架代码更新可能导致的**模型训练、预测报错、性能下降**等问题。本文主要介绍训推一体中**Paddle2ONNX预测链条**的接入规范和监测点，是在[Linux GPU/CPU 基础训练推理测试开发规范](http://agroup.baidu.com/paddlepaddle/md/article/4273691)上针对Paddle2ONNX链条的补充说明。

主要监控的内容有：

- 框架更新后，套件模型的 Paddle2ONNX 预测是否能正常走通；（比如 API 的不兼容升级）

为了能监控上述问题，希望把套件模型的 Paddle2ONNX 预测链条加到框架的 CI 和 CE 中，提升 PR 合入的质量。因此，需要在套件中加入运行脚本（不影响套件正常运行），完成模型的自动化测试。

可以建立的 CI/CE 机制包括：

 1. 不训练，全量数据走通开源模型 Paddle2ONNX 预测，并验证模型预测精度是否符合设定预期；（单模型30分钟内） [保证] **
	 a. 保证 Paddle2ONNX 模型转化走通，可以正常通过onnxruntime部署，得到正确预测结果（QA添加中）

注：由于 CI 有时间限制，所以在测试的时候需要限制运行时间，所以需要构建一个很小的数据集完成测试。

<a name="12"></a>
## 1.2 训推一体自动化测试

本规范测试的链条如下，可以根据模型开发规范（ http://agroup.baidu.com/paddlepaddle/md/article/3638870 ）和套件需要，适当删减链条。
![图片](http://bos.bj.bce-internal.sdns.baidu.com/agroup-bos-bj/bj-6c2fa95d0fb01317a5858c81c53dfb1ba92a5f72)

上图各模块具体测试点如下：

- 服务化模型转换 （必选）
- 部署环境软硬件选型
	- onnxruntime **（必选）**
		- CPU（Linux **必选** ，MacOS 可选，Windows 可选）
	- onnxruntime-gpu（可选）
		- GPU （Linux 可选， MacOS 可选， Windows 可选）

本文档目前只支持了必选链条，可选模块后续完善。

<a name="13"></a>
## 1.3 文本检测样板间概览

在 PaddleOCR 中，以文本检测为例，提供了本规范的样板间，可以跑通1.2章节提到的**所有测试链条**，完成1.1背景部分提到的1种 CI/CE 机制。

Paddle2ONNX 链条测试工具已与基础链条测试集成到一起，位于 PaddleOCR dygraph 分支下的[test_tipc目录](https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph/test_tipc)，其关键结构如下：
```

test_tipc/
├── common_func.sh
├── configs
│   ├── ppocr_det_model
│   │   ├── model_linux_gpu_normal_normal_paddle2onnx_python_linux_gpu_cpu.txt # 测试OCR检测模型的参数配置文件
│   ├── xxx
├── docs                              # 各测试流程文档说明
│   ├── test_paddle2onnx.md              # Paddle2ONNX 部署测试脚本运行说明
│   ├── xxx
├── output
├── prepare.sh                        # 完成训推一体运行所需要的数据和模型下载
├── readme.md                         # 飞桨训推一体认证说明            
├── test_paddle2onnx.sh                   # Paddle2ONNX 部署启动脚本
├── xxx
...
```

<a name="2"></a>
# 2. 训推一体规范接入 Paddle2ONNX 预测流程
训推一体规范接入包含如下三个步骤，接下来将依次介绍这三个部分。  

 - 准备数据和环境
 - 规范化输出日志
 - 编写自动化测试代码

<a name="21"></a>
## 2.1 准备数据和环境

同标准训推一体测试流程一样，在 prepare.sh 中准备好所需数据和环境，包括：

- 少量预测数据
- inference 预测模型
- Paddle2ONNX 所需 whl 包

以 PaddleOCR 的检测模型为例，使用方式：
```
#                     配置文件路径                 运行模式
bash tests/prepare.sh ./tests/model_linux_gpu_normal_normal_paddle2onnx_python_linux_gpu_cpu 'paddle2onnx_infer'
```
prepare.sh 具体内容：

```
# 判断预测模式
#!/bin/bash
FILENAME=$1

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer', 'cpp_infer', 'serving_infer', 'paddle2onnx_infer']
MODE=$2

.....


if [ ${MODE} = "paddle2onnx_infer" ];then
    # 准备 paddle2onnx 环境
    python_name=$(func_parser_value "${lines[2]}")
    ${python_name} -m pip install install paddle2onnx
    ${python_name} -m pip install onnxruntime==1.4.0
    # 下载 paddle inference 模型
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar
    wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar
    # 下载预测数据
    wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/ch_det_data_50.tar
    wget -nc -P ./inference/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/rec_inference.tar
    cd ./inference && tar xf ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_ppocr_server_v2.0_rec_infer.tar && tar xf ch_ppocr_server_v2.0_det_infer.tar && tar xf ch_det_data_50.tar && tar xf rec_inference.tar && cd ../
    
fi
```

<a name="22"></a>
## 2.2 规范化输出预测日志
<a name="221"></a>
### 2.2.1 预测日志规范
（1）背景
类似于 python 预测等基础测试链条，Paddle2ONNX 预测链条也需要规范不同套件中 paddle inference 预测输出的格式，方便统一自动化测试。由于Paddle2ONNX启动的特殊性，该链条只监控预测正确性，性能数据暂不进行监控。

Paddle2ONNX 测试要求规范输出预测结果及以下信息：

- 运行的硬件，CPU、GPU、XPU、Lite
- 运行的模型名称
- 运行的数据信息，包括 batch size，数据量
- 图片的预测结果

<a name="222"></a>
### 2.2.2 接入步骤

Paddle2ONNX 测试链条无需接入 AutoLog 工具包，注意日志导出名称需符合规范，具体在编写自动化测试代码中说明。


<a name="223"></a>
## 2.3 编写自动化测试代码

如果已经完成 python 预测链条的接入，那么 Paddle2ONNX 链条接入是类似的。

自动化测试脚本包括三个部分，分别是运行脚本`test_paddle2onnx.sh`，参数文件`model_linux_gpu_normal_normal_paddle2onnx_python_linux_gpu_cpu.txt`，数据模型准备脚本`prepare.sh`。理论上只需要修改`model_linux_gpu_normal_normal_paddle2onnx_python_linux_gpu_cpus.txt`和`prepare.sh`就可以完成自动化测试，本节将详细介绍如何修改`model_linux_gpu_normal_normal_paddle2onnx_python_linux_gpu_cpu.txt`，完成 Paddle2ONNX 预测测试。运行脚本test_paddle2onnx.sh将会在附录中详细介绍。

按如下方式在参数文件`model_linux_gpu_normal_normal_paddle2onnx_python_linux_gpu_cpu.txt`中添加 Paddle2ONNX 预测部分参数：

![图片](http://bos.bj.bce-internal.sdns.baidu.com/agroup-bos-bj/bj-de608be2b7018147f4cceedeb325cf39f228a6e1)
参数说明：

|行号 | 参数 | 参数介绍 | 
|---|---|---|
|1 | model_name | 模型名称 |
|2 | python | python版本 |
|3 | 2onnx: paddle2onnx | paddle2onnx 命令|
|4 | --model_dir:./inference/ch_ppocr_mobile_v2.0_det_infer/ | inference 模型保存路径 |
|5 | --model_filename:inference.pdmodel| pdmodel 文件名 |
|6 | --params_filename:inference.pdiparams | pdiparams 文件名 |
|7 | --save_file:./inference/det_mobile_onnx/model.onnx | 转换出的 onnx 模型目录|
|8 | --opset_version:10 | onnx op 版本 | 
|9 | --enable_onnx_checker:True | 是否进行模型转换检测 |
|10 | inference:tools/infer/predict_det.py | 启动 inference 预测命令 |
|11 | --use_gpu:True\|False | 是否使用GPU |
|12 | --det_model_dir: | 检测模型地址（默认为空，表示与转换出的onnx模型目录一致） | 
|13 | --image_dir:./inference/ch_det_data_50/all-sum-510/  | 预测图片路径 |


<a name="3"></a>
# 3. 附录
<a name="31"></a>
## 3.1 自动化测试脚本test_paddle2onnx.sh 函数介绍
Paddle2ONNX 预测核心函数：

- func_paddle2onnx()：执行 Paddle2ONNX 预测的函数，由于主要测试的功能为模型转换是否正确，因此目前的测试环境仅有CPU。

以下 function 和 python 预测链条功能一致。

- func_parser_key() ：解析params.txt中：前的部分

- func_parser_value() ：解析params.txt中：后的部分

- func_set_params()  ：会返回 key=value 的字符串，用与组建参数命令，如果key或者value为null，则返回空字符，即这个参数不会生效

- func_parser_params() ：为了解析和测试模式 MODE 相关的参数，目前只用来解析 epoch 和 batch size 参数

- status_check() ：状态检查函数，获取上条指令运行的状态，如果是0，则运行成功，如果是其他则运行失败，失败和成功的指令都会存放在 results.log 文件中

<a name="32"></a>
## 3.2 其他说明

test_paddle2onnx.sh 是兼容性修改，理论上所有套件用一套代码，test_paddle2onnx.sh 中也做了很多兼容性的改动，如果还有哪些兼容性的问题需要改进，包括使用问题可以提交Issue。
