# 10分钟快速上手PaddleCV

PaddleCV是飞桨视觉统一的推理部署套件，提供了单模型、多模型串联部署流程。本章节我们将详细讲解PaddleCV使用方法

- [安装](#1)
- [预测部署](#2)
  - [部署示例](#2.1)
  - [参数说明](#2.2)
  - [配置文件](#2.3)
- [二次开发](#3)

<a name="1"></a>

## 1. 安装

关于安装配置运行环境，请参考[安装指南](INSTALL.md)

<a name="2"></a>

## 2. 预测部署

PaddleCV预测部署依赖推理模型（paddle.jit.save保存的模型），PaddleCV的配置文件中预置了不同任务推荐的推理模型下载链接。如果需要依赖飞桨各开发套件进行二次开发，相应导出文档链接如下表所示

| 开发套件名称   |  导出模型文档链接 |
|:-----------:|:------------------:|
|  PaddleClas    |              [文档链接](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/deployment/export_model.md)                      |
|  PaddleDetection    |        [文档链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/EXPORT_MODEL.md)                     |
|  PaddleSeg    |              [文档链接](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/model_export_cn.md)                       |
|  PaddleOCR    |              [文档链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/table_recognition.md#41-%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA)                       |

注意：

1. PaddleOCR分别提供了不同任务的导出模型方法，上表提供链接为文本检测模型导出文档，其他任务可以参考[文档教程](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/README_ch.md#-%E6%96%87%E6%A1%A3%E6%95%99%E7%A8%8B)

<a name="2.1"></a>

### 1）部署示例

得到导出模型后可以使用如下命令进行预测部署：

```bash
# 图像分类任务
python -u tools/predict.py --config=configs/single_op/PP-HGNet.yml --input=demo/ILSVRC2012_val_00020010.jpeg

# 目标检测任务
python -u tools/predict.py --config=configs/single_op/PP-YOLOE+.yml --input=demo/000000014439.jpg

# 图像分割任务
python -u tools/predict.py --config=configs/single_op/PP-HumanSegV2.yml --input=demo/pp_humansegv2_demo.jpg

# OCR任务
python -u tools/predict.py --config=configs/system/PP-OCRv3.yml --input=demo/word_1.jpg
```

使用whl包安装后，也可以在python中使用三行代码快速进行预测部署，示例如下：

```python
from paddlecv import PaddleCV
paddlecv = PaddleCV(task_name="PP-OCRv3")
res = paddlecv("../demo/00056221.jpg")
```

<a name="2.2"></a>

### 2）参数说明

| 参数名   |  是否必选 | 默认值 | 含义 |
|:------:|:---------:|:---------:|:---------:|
|  config    |     是   |   None |  配置文件路径          |
|  input    |     是   |   None |  输入路径，支持图片文件，图片文件夹和视频文件          |
|  output_dir    |     否   |   output |  输出结果保存路径，包含可视化结果和结构化输出          |
|  run_mode    |     否   |   paddle |  预测部署模式，可选项为`'paddle'/'trt_fp32'/'trt_fp16'/'trt_int8'/'mkldnn'/'mkldnn_bf16'`    |
|  device    |     否   |   CPU |  运行设备，可选项为`CPU/GPU/XPU`      |

<a name="2.3"></a>

### 3）配置文件

配置文件划分为[单模型配置](../configs/single_op)和[串联系统配置](../configs/system)。配置内容分类环境类配置和模型类配置。环境配置中包含device设置，输出结果保存路径等字段。模型配置中包含各个模型的预处理，模型推理，输出后处理全流程配置项。需要注意使用正确的`param_path`和`model_path`路径。具体配置含义可以参考[配置文件说明文档](config_anno.md)

支持通过命令行修改配置文件内容，示例如下：

```
# 通过-o修改检测后处理阈值
python -u tools/predict.py --config=configs/single_op/PP-YOLOE+.yml --input=demo/000000014439.jpg -o MODEL.0.DetectionOp.PostProcess.0.ParserDetResults.threshold=0.6
```

**注意：**

1. 优先级排序：命令行输入 > 配置文件配置
2. -o 中的`0`表示MODEL的Op位置，防止模型串联过程中出现同类Op的情况

<a name="3"></a>

## 3. 二次开发

PaddleCV中内置了分类、检测、分割等单模型算子，以及OCR，行人分析工具等串联系统算子。如果用户在使用过程中需要自定义算子进行二次开发，可以参考[新增算子文档](how_to_add_new_op.md)和[外部算子开发文档](custom_ops.md)
