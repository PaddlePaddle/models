# 新增算子

## 1. 简介

本教程主要介绍怎样基于PaddleCV新增推理算子。

本项目中，算子主要分为3个部分。

- 模型推理算子：给定输入，加载模型，完成预处理、推理、后处理，返回输出。
- 模型衔接算子：给定输入，计算得到输出。一般用于将一个模型的输出处理为另外一个模型的输入，比如说目标检测、文本检测的扣图、方向矫正模块之后的模型旋转、文本合成等操作。
- 模型输出算子：存储、可视化、输出模型的输出结果。

在下面的介绍中，我们把算子称为op。


## 2. 单个op的输入/输出格式

PaddleCV的输入为图像或者视频。

对于所有的op，系统会将其整理为`a list of dict`的格式。列表中的每个元素均为一个待推理的对象及其中间结果。比如，对于图像分类来说，其输入仅包含图像信息，输入格式如下所示。


```json
[
    {"image": img1},
    {"image": img2},
]
```


输出格式为

```json
[
    {"image": img1, "class_ids": class_id1, "scores": scores1, "label_names": label_names1},
    {"image": img2, "class_ids": class_id2, "scores": scores2, "label_names": label_names2},
]
```

同理，对于模型衔接算子（BBoxCropOp为例）来说，其输入如下。

```json
[
    {"image": img1, "bbox": bboxes1},
    {"image": img2, "bbox": bboxes2},
]
```


## 3. 新增算子

### 3.1 模型推理算子

模型推理算子，整体继承自[ModelBaseOp类](../ppcv/ops/models/base.py)。示例可参考图像分类op：[ClassificationOp类](../ppcv/ops/models/classification/inference.py)。具体地，我们需要实现以下几个内容。

（1）该类需要继承自`ModelBaseOp`，同时使用`@register`方法进行注册，保证全局唯一。

（2）实现类中一些方法，包括

- 初始化`__init__`
    - 输入：model_cfg与env_cfg
    - 输出：无
- 模型预处理`preprocess`
    - 输入：基于input_keys过滤后的模型输入
    - 输出：模型预处理结果
- 模型后处理`postprocess`
    - 输入：模型推理结果
    - 输出：模型后处理结果
- 预测`__call__`
    - 输入：该op依赖的输入内容
    - 输出：该op的处理结果



### 3.2 模型衔接算子


模型衔接算子，整体继承自[ConnectorBaseOp](../ppcv/ops/connector/base.py)。示例可参考方向矫正op：[ClsCorrectionOp类](../ppcv/ops/connector/op_connector.py)。具体地，我们需要实现以下几个内容。

（1）该类需要继承自`ConnectorBaseOp`，同时使用`@register`方法进行注册，保证全局唯一。

（2）实现类中一些方法，包括

- 初始化`__init__`
    - 输入：model_cfg、env_cfg(一般为None)
    - 输出：无
- 调用`__call__`
    - 输入：该op依赖的输入内容
    - 输出：该op的处理结果


### 3.3 模型输出算子


模型衔接算子，整体继承自[OutputBaseOp](../ppcv/ops/output/base.py)。示例可参考方向矫正op：[ClasOutput类](../ppcv/ops/output/classification.py)。具体地，我们需要实现以下几个内容。

（1）该类需要继承自`OutputBaseOp`，同时使用`@register`方法进行注册，保证全局唯一。

（2）实现类中一些方法，包括

- 初始化`__init__`
    - 输入：model_cfg、env_cfg(一般为None)
    - 输出：无
- 调用`__call__`
    - 输入：模型输出
    - 输出：返回结果


## 4. 新增单测

在新增op之后，需要新增基于该op的单测，可以参考[test_classification.py](../tests/test_classification.py)。
