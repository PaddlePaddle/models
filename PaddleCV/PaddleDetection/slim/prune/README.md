>运行该示例前请安装Paddle1.6或更高版本

# 检测模型卷积通道剪裁示例

## 概述

该示例使用PaddleSlim提供的[卷积通道剪裁压缩策略](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/tutorial.md#2-%E5%8D%B7%E7%A7%AF%E6%A0%B8%E5%89%AA%E8%A3%81%E5%8E%9F%E7%90%86)对检测库中的模型进行压缩。
在阅读该示例前，建议您先了解以下内容：

- <a href="../..README_cn.md">检测库的常规训练方法</a>
- [PaddleSlim使用文档](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md)


## 配置文件说明

关于配置文件如何编写您可以参考：

- [PaddleSlim配置文件编写说明](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md#122-%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E7%9A%84%E4%BD%BF%E7%94%A8)
- [裁剪策略配置文件编写说明](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md#22-%E6%A8%A1%E5%9E%8B%E9%80%9A%E9%81%93%E5%89%AA%E8%A3%81)

其中，配置文件中的`pruned_params`需要根据当前模型的网络结构特点设置，它用来指定要裁剪的parameters.

这里以MobileNetV1-YoloV3模型为例，其卷积可以三种：主干网络中的普通卷积，主干网络中的`depthwise convolution`和`yolo block`里的普通卷积。PaddleSlim暂时无法对`depthwise convolution`直接进行剪裁， 因为`depthwise convolution`的`channel`的变化会同时影响到前后的卷积层。我们这里只对主干网络中的普通卷积和`yolo block`里的普通卷积做裁剪。

通过以下方式可视化模型结构：

```
from paddle.fluid.framework import IrGraph
from paddle.fluid import core

graph = IrGraph(core.Graph(train_prog.desc), for_test=True)
marked_nodes = set() 
for op in graph.all_op_nodes():
    print(op.name())
    if op.name().find('conv') > -1:
        marked_nodes.add(op)
graph.draw('.', 'forward', marked_nodes)
```

该示例中MobileNetV1-YoloV3模型结构的可视化结果：<a href="./images/MobileNetV1-YoloV3.pdf">MobileNetV1-YoloV3.pdf</a>

同时通过以下命令观察目标卷积层的参数（parameters）的名称和shape: 

```
for param in fluid.default_main_program().global_block().all_parameters():
    if 'weights' in param.name:
        print param.name, param.shape
```


从可视化结果，我们可以排除后续会做concat的卷积层，总终得到如下要裁剪的参数名称：

```
conv2_1_sep_weights
conv2_2_sep_weights
conv3_1_sep_weights
conv4_1_sep_weights
conv5_1_sep_weights
conv5_2_sep_weights
conv5_3_sep_weights
conv5_4_sep_weights
conv5_5_sep_weights
conv5_6_sep_weights
yolo_block.0.0.0.conv.weights
yolo_block.0.0.1.conv.weights
yolo_block.0.1.0.conv.weights
yolo_block.0.1.1.conv.weights
yolo_block.1.0.0.conv.weights
yolo_block.1.0.1.conv.weights
yolo_block.1.1.0.conv.weights
yolo_block.1.1.1.conv.weights
yolo_block.1.2.conv.weights
yolo_block.2.0.0.conv.weights
yolo_block.2.0.1.conv.weights
yolo_block.2.1.1.conv.weights
yolo_block.2.2.conv.weights
yolo_block.2.tip.conv.weights
```

```
(conv2_1_sep_weights)|(conv2_2_sep_weights)|(conv3_1_sep_weights)|(conv4_1_sep_weights)|(conv5_1_sep_weights)|(conv5_2_sep_weights)|(conv5_3_sep_weights)|(conv5_4_sep_weights)|(conv5_5_sep_weights)|(conv5_6_sep_weights)|(yolo_block.0.0.0.conv.weights)|(yolo_block.0.0.1.conv.weights)|(yolo_block.0.1.0.conv.weights)|(yolo_block.0.1.1.conv.weights)|(yolo_block.1.0.0.conv.weights)|(yolo_block.1.0.1.conv.weights)|(yolo_block.1.1.0.conv.weights)|(yolo_block.1.1.1.conv.weights)|(yolo_block.1.2.conv.weights)|(yolo_block.2.0.0.conv.weights)|(yolo_block.2.0.1.conv.weights)|(yolo_block.2.1.1.conv.weights)|(yolo_block.2.2.conv.weights)|(yolo_block.2.tip.conv.weights)
```

综上，我们将MobileNetV2配置文件中的`pruned_params`设置为以下正则表达式：

```
(conv2_1_sep_weights)|(conv2_2_sep_weights)|(conv3_1_sep_weights)|(conv4_1_sep_weights)|(conv5_1_sep_weights)|(conv5_2_sep_weights)|(conv5_3_sep_weights)|(conv5_4_sep_weights)|(conv5_5_sep_weights)|(conv5_6_sep_weights)|(yolo_block.0.0.0.conv.weights)|(yolo_block.0.0.1.conv.weights)|(yolo_block.0.1.0.conv.weights)|(yolo_block.0.1.1.conv.weights)|(yolo_block.1.0.0.conv.weights)|(yolo_block.1.0.1.conv.weights)|(yolo_block.1.1.0.conv.weights)|(yolo_block.1.1.1.conv.weights)|(yolo_block.1.2.conv.weights)|(yolo_block.2.0.0.conv.weights)|(yolo_block.2.0.1.conv.weights)|(yolo_block.2.1.1.conv.weights)|(yolo_block.2.2.conv.weights)|(yolo_block.2.tip.conv.weights)
```

我们可以用上述操作观察其它检测模型的参数名称规律，然后设置合适的正则表达式来剪裁合适的参数。

## 训练

根据<a href="../../tools/train.py">PaddleDetection/tools/train.py</a>编写压缩脚本compress.py。
在该脚本中定义了Compressor对象，用于执行压缩任务。


您可以通过运行脚本`run.sh`运行该示例。


### 保存断点（checkpoint）

如果在配置文件中设置了`checkpoint_path`, 则在压缩任务执行过程中会自动保存断点，当任务异常中断时，
重启任务会自动从`checkpoint_path`路径下按数字顺序加载最新的checkpoint文件。如果不想让重启的任务从断点恢复，
需要修改配置文件中的`checkpoint_path`，或者将`checkpoint_path`路径下文件清空。

>注意：配置文件中的信息不会保存在断点中，重启前对配置文件的修改将会生效。


## 评估

如果在配置文件中设置了`checkpoint_path`，则每个epoch会保存一个压缩后的用于评估的模型，
该模型会保存在`${checkpoint_path}/${epoch_id}/eval_model/`路径下，包含`__model__`和`__params__`两个文件。
其中，`__model__`用于保存模型结构信息，`__params__`用于保存参数（parameters）信息。

如果不需要保存评估模型，可以在定义Compressor对象时，将`save_eval_model`选项设置为False（默认为True）。

## 预测

如果在配置文件中设置了`checkpoint_path`，并且在定义Compressor对象时指定了`prune_infer_model`选项，则每个epoch都会
保存一个`inference model`。该模型是通过删除eval_program中多余的operators而得到的。

该模型会保存在`${checkpoint_path}/${epoch_id}/eval_model/`路径下，包含`__model__.infer`和`__params__`两个文件。
其中，`__model__.infer`用于保存模型结构信息，`__params__`用于保存参数（parameters）信息。

更多关于`prune_infer_model`选项的介绍，请参考：[Compressor介绍](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md#121-%E5%A6%82%E4%BD%95%E6%94%B9%E5%86%99%E6%99%AE%E9%80%9A%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC)

### python预测

在脚本<a href="../../tools/infer.py">PaddleDetection/tools/infer.py</a>中展示了如何使用fluid python API加载使用预测模型进行预测。

### PaddleLite

该示例中产出的预测（inference）模型可以直接用PaddleLite进行加载使用。
关于PaddleLite如何使用，请参考：[PaddleLite使用文档](https://github.com/PaddlePaddle/Paddle-Lite/wiki#%E4%BD%BF%E7%94%A8)

## 示例结果

### MobileNetV1-YOLO-V3

| FLOPS |top1_acc/top5_acc| model_size |Paddle Fluid inference time(ms)| Paddle Lite inference time(ms)|
|---|---|---|---|---|
|baseline|- |- |- |-|
|-10%|- |- |- |-|
|-30%|- |- |- |-|
|-50%|- |- |- |-|

## FAQ
