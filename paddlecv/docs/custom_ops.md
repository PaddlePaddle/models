# 外部算子开发

- [简介](#1)
- [外部算子依赖](#2)
- [外部算子实现方式](#3)


<a name="1"></a>

## 1. 简介

本教程主要介绍基于paddlecv新增外部算子，实现定制化算子开发，进行外部算子开发前，首先准备paddlecv环境，推荐使用pip安装

```bash
pip install paddlecv
```

<a name="2"></a>

## 2. 外部算子依赖

外部算子主要依赖接口如下：

#### 1）`ppcv.ops.base.create_operators(params, mod)`

  - 功能：创建预处理后处理算子接口
  - 输入：
    - params: 前后后处理配置字典
    - mod: 当前算子module
  - 输出：前后处理算子实例化对象列表



#### 2）算子BaseOp

外部算子类型和paddlecv内算子类型相同，分为模型算子、衔接算子和输出算子。新增外部算子需要继承每类算子对应的BaseOp，对应关系如下：

 ```txt
 模型算子：ppcv.ops.models.base.ModelBaseOp
 衔接算子：ppcv.ops.connector.base.ConnectorBaseOp
 输出算子：ppcv.ops.output.base.OutputBaseOp
 ```

#### 3）ppcv.core.workspace.register

需要使用@register对每个外部算子类进行修饰，例如：

```python
from ppcv.ops.models.base import ModelBaseOp
from ppcv.core.workspace import register

@register
class DetectionCustomOp(ModelBaseOp)
```

<a name="3"></a>

## 3. 外部算子实现方式

可直接参考[新增算子文档](how_to_add_new_op.md)，实现后使用方式与paddlecv内部提供算子相同。paddlecv中提供检测外部算子[示例](../custom_op)
