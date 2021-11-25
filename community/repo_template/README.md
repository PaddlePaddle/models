# 论文名称

## 目录

```
1. 简介
2. 数据集和复现精度
3. 开始使用
4. 代码结构与详细说明
```

**注意：** 目录可以使用[gh-md-toc](https://github.com/ekalinin/github-markdown-toc)生成

## 1. 简介

简单的介绍模型，以及模型的主要架构或主要功能，如果能给出效果图，可以在简介的下方直接贴上图片，展示模型效果。然后另起一行，按如下格式给出论文名称及链接、参考代码链接、aistudio体验教程链接。

注意：在给出参考repo的链接之后，建议添加对参考repo的开发者的致谢。

**论文:** [title](url)

**参考repo:** [repo name](url)

在此非常感谢`$参考repo的 github id$`等人贡献的[repo name](url)，提高了本repo复现论文的效率。

**aistudio体验教程:** [地址](url)


## 2. 数据集和复现精度

给出本repo中用到的数据集的链接，然后按格式描述数据集大小与数据集格式。

格式如下：

- 数据集大小：关于数据集大小的描述，如类别，数量，图像大小等等
- 数据集下载链接：链接地址
- 数据格式：关于数据集格式的说明

基于上述数据集，给出论文中精度、参考代码的精度、本repo复现的精度、数据集名称、模型下载链接（模型权重和对应的日志文件推荐放在**百度云网盘**中，方便下载）、模型大小，以表格的形式给出。如果超参数有差别，可以在表格中新增一列备注一下。

如果涉及到`轻量化骨干网络验证`，需要新增一列骨干网络的信息。



## 3. 开始使用

### 3.1 准备环境

首先介绍下支持的硬件和框架版本等环境的要求，格式如下：

- 硬件：xxx
- 框架：
  - PaddlePaddle >= 2.1.0

然后介绍下怎样安装PaddlePaddle以及对应的requirements。

建议将代码中用到的非python原生的库，都写在requirements.txt中，在安装完PaddlePaddle之后，直接使用`pip install -r requirements.txt`安装依赖即可。


### 3.2 快速开始

需要给出快速训练、预测、使用预训练模型预测、模型导出、模型基于inference模型推理的使用说明，同时基于demo图像，给出预测结果和推理结果，并将结果打印或者可视化出来。

## 4. 代码结构与详细说明

### 4.1 代码结构

需要用一小节描述整个项目的代码结构，用一小节描述项目的参数说明，之后各个小节详细的描述每个功能的使用说明。

### 4.2 参数说明

以表格的形式，给出当前的参数列表、含义、类型、默认值等信息。

### 4.3 基础使用

配合部分重要配置参数，介绍模型训练、评估、预测、导出等过程。

### 4.4 模型部署

给出当前支持的推理部署方式以及相应的参考文档链接。

### 4.5 TIPC测试支持

这里需要给出TIPC的目录链接。

**注意：** 这里只需提供TIPC基础测试链条中模式`lite_train_lite_infer`的代码与文档即可。


* 更多关于TIPC的介绍可以参考：[飞桨训推一体认证（TIPC）文档](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/test_tipc/readme.md)
* 关于Linux端基础链条测试接入的代码与文档说明可以参考：[基础链条测试接入规范](https://github.com/PaddlePaddle/models/blob/tipc/docs/tipc_test/development_specification_docs/train_infer_python.md)，[PaddleOCR Linux端基础训练预测功能测试文档](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/test_tipc/docs/test_train_inference_python.md)


如果您有兴趣，也欢迎为项目集成更多的TIPC测试链条及相关的代码文档，非常感谢您的贡献。
