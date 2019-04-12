<div align="center">
  <h3>
    <a href="tutorial.md">
      算法原理介绍
    </a>
    <span> | </span>
    <a href="demo.md">
      示例文档
    </a>
    <span> | </span>
    <a href="model_zoo.md">
      Model Zoo
    </a>
  </h3>
</div>


---
# Paddle模型压缩工具库使用说明

本文第一章介绍PaddleSlim模块通用功能的使用，不涉及具体压缩策略的细节。第二、三、四章分别介绍量化训练、剪切、蒸馏三种压缩策略的使用方式。
建议在看具体策略使用方式之前，先浏览下对应的原理介绍：<a href="tutorial.md">算法原理介绍</a>

>在本文中不区分operator和layer的概念。不区分loss和cost的概念。

## 目录

- [通用功能使用说明](#1-paddleslim通用功能使用介绍)
- [量化使用说明](#21-量化训练)
- [剪切使用说明](#22-卷积核剪切)
- [蒸馏使用说明](#23-蒸馏)


## 1. PaddleSlim通用功能使用介绍

## 1.1 使用压缩工具库的前提

### 1.1.1 安装paddle

**版本：** PaddlePaddle >= 1.4
**安装教程：** [安装说明](http://paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/index_cn.html)


### 1.1.2 搭建好网络结构

用户需要搭建好前向网络，并可以正常执行。
一个正常可执行的网络一般需要以下内容或操作：

- 网络结构的定义
- data_reader
- optimizer
- 初始化，load pretrain model
- feed list与fetch list

#### 1.1.2.1 网络结构的定义
首先参考以下文档，配置网络：
[《Paddle使用指南：配置简单的网络》](http://paddlepaddle.org/documentation/docs/zh/1.3/user_guides/howto/configure_simple_model/index.html)

这一步的产出应该是两个[Program](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/fluid_cn.html#program)实例：

- **train_program:** 用于在压缩过程中迭代训练模型，该program必须包含loss。一般改program不要有backward op和weights update op，否则不能使用蒸馏策略。

- **eval_program:** 用于在压缩过程中评估模型的精度，一般会包含accuracy、IoU等评估指标的计算layer。

>在量化训练策略中，会根据eval_program进行网络结构剪枝并保存一个用于inference的量化模型。这时候，就要求inference网络是eval_program的一个子网络。

#### 1.1.2.2. data_reader

按照以下文档准备数据：
[《Paddle使用指南：准备数据》](http://paddlepaddle.org/documentation/docs/zh/1.3/user_guides/howto/prepare_data/index.html)

这一步需要产出两个DataReader:

**train_reader:** 用于给train_program的执行提供数据
**eval_reader:** 用于给eval_program的执行提供数据

#### 1.1.2.3. optimizer
[fluid.optimizer API](http://www.paddlepaddle.org/documentation/docs/zh/1.3/api_cn/optimizer_cn.html)

在不同的使用场景下，用户需要提供0个、1个或2两个optimizer:

- **0个optimizer:** 在模型搭建阶段的train_program已经是一个包含了反向op和模型weight更新op的网络，则不用再提供optimizer
- **1个optimizer:** train_program只有前向计算op, 则需要提供一个optimizer，用于优化训练train_program.
-  **2个optimizer:** 在使用蒸馏策略时，且蒸馏训练阶段和单独fine-tune阶段用不同的优化策略。一个optimizer用于优化训练teacher网络和student网络组成的蒸馏训练网络，另一个optimizer用于单独优化student网络。更多细节会在蒸馏策略使用文档中介绍。

#### 1.1.2.4. load pretrain model

- 剪切：需要加载pretrain model
- 蒸馏：根据需要选择是否加载pretrain model
- 量化训练：需要加载pretrain model

#### 1.1.2.5. feed list与fetch list
feed list和fetch list是两个有序的字典, 示例如下：
```
feed_list = [('image', image.name), ('label', label.name)]
fetch_list = [('loss', avg_cost.name)]
```
其中，feed_list中的key为自定义的有一定含义的字符串，value是[Variable](http://paddlepaddle.org/documentation/docs/zh/1.3/api_guides/low_level/program.html#variable)的名称, feed_list中的顺序需要和DataReader提供的数据的顺序对应。

对于train_program和eval_program都需要有与其对应的feed_list和fetch_list。

>注意： 在train_program对应的fetch_list中，loss variable(loss layer的输出)对应的key一定要是‘‘loss’’


## 1.2 压缩工具库的使用

经过1.1节的准备，所以压缩工具用到的关于目标模型的信息已经就绪，执行以下步骤配置并启动压缩任务：

- 改写模型训练脚本，加入模型压缩逻辑
- 编写配置文件
- 执行训练脚本进行模型压缩

### 1.2.1 如何改写普通训练脚本

在1.1节得到的模型脚本基础上做如下修改：

第一步： 构造`paddle.fluid.contrib.slim.Compressor`对象, Compressor构造方法参数说明如下：

```
Compressor(place,
             scope,
             train_program,
             train_reader=None,
             train_feed_list=None,
             train_fetch_list=None,
             eval_program=None,
             eval_reader=None,
             eval_feed_list=None,
             eval_fetch_list=None,
             teacher_programs=[],
             checkpoint_path='./checkpoints',
             train_optimizer=None,
             distiller_optimizer=None)
```
- **place:**  压缩任务使用的device。GPU请使用[paddle.fluid.CUDAPlace(0)](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/fluid_cn.html#paddle.fluid.CUDAPlace)
- **scope:** 如果在网络配置阶段没有构造scope，则用的是[global scope](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/executor_cn.html#paddle.fluid.global_scope)，该参数设置为`paddle.fluid.global_scope()`. 如果有自己构造scope，则设置为自己构造的scope.
- **train_program:** 该program内的网络只有前向operator，而且必须带有loss. 关于program的概念，请参考：[Program API](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/fluid_cn.html#program)
- **train_reader:** 提供训练数据的[data reader](http://paddlepaddle.org/documentation/docs/zh/1.3/user_guides/howto/prepare_data/reader_cn.html)
- **train_feed_list:** 用于指定train program的输入节点, 详见：1.1.2.5节。
- **train_fetch_list:** 用于指定train program的输出节点，详见：1.1.2.5节。
- **eval_program:** 用于评估模型精度的program
- **eval_reader:** 提供评估数据的[data reader](http://paddlepaddle.org/documentation/docs/zh/1.3/user_guides/howto/prepare_data/reader_cn.html)
- **eval_feed_list:** 用于指定eval program的输入节点，详见：1.1.2.5节。
- **eval_fetch_list:** 用于指定eval program的输出节点, 格式同train_fetch_list, 详见：1.1.2.5节。
- **teacher_programs:** 用于蒸馏的programs, 这些program需要和train program共用同一个scope.
- **train_optimizer:** 用于训练train program的优化器
- **distiller_optimizer:** 用于蒸馏训练的优化器


第2步：读取配置文件和调用run方法，示例如下
```python
compressor.config('./compress.yaml')
compressor.run()
```
其中，compress.yaml文件是压缩策略配置文件，集中了压缩策略的所有可调节参数，在1.2.2节中会详细介绍其格式和内容。

完成该节操作后的完整示例见：[compress.py]()

### 1.2.2 配置文件的使用

模型压缩模块用[yaml](https://zh.wikipedia.org/wiki/YAML)文件集中管理可调节的压缩策略参数。我们以filter pruning为例，说明配置文件的编写方式。

第一步：注册pruners, 如下所示，指定pruner的类别和一些属性，后文**第5节**会详细介绍可选类别和属性的含义。
```python
pruners:
    pruner_1:
        class: 'StructurePruner'
        pruning_axis:
            '*': 0
        criterions:
            '*': 'l1_norm'
```

第二步：注册剪切策略
如下所示，我们注册两个uniform剪切策略，分别在第0个epoch和第10个epoch将模型的FLOPS剪掉10%.
```python
strategies:
    pruning_strategy_0:
        class: 'UniformPruneStrategy'
        pruner: 'pruner_1'
        start_epoch: 0
        target_ratio: 0.10
        pruned_params: '.*_sep_weights'
        metric_name: 'acc_top1'
    pruning_strategy_1:
        class: 'UniformPruneStrategy'
        pruner: 'pruner_1'
        start_epoch: 10
        target_ratio: 0.10
        pruned_params: '.*_sep_weights'
        metric_name: 'acc_top1'
```

第三步：配置通用参数

我们在compress_pass下配置整个压缩任务的参数，如下所示，整个压缩任务会执行120个epoch, 压缩过程中的checkpoint保存在./checkpoints路径下。compress_pass.strategies下为生效的压缩策略，如果生效的多个策略的start_epoch参数一样，则按compress_pass.strategies下列出的先后顺序被调用。

```python
compress_pass:
    epoch: 120
    checkpoint_path: './checkpoints/'
    strategies:
        - pruning_strategy_0
        - pruning_strategy_1
```


## 2. 模型压缩策略使用介绍

本章依次介绍量化训练、卷积核剪切和蒸馏三种策略的使用方式，在此之前建议先浏览相应策略的原理介绍：

- [量化训练原理](tutorial.md#1-quantization-aware-training量化介绍)
- [卷积核剪切原理](tutorial.md#2-卷积核剪切原理)
- [蒸馏原理](tutorial.md#3-蒸馏)

### 2.1 量化训练

**用户须知:** 现阶段的量化训练主要针对卷积层（包括二维卷积和Depthwise卷积）以及全连接层进行量化。卷积层和全连接层在PaddlePaddle框架中对应算子包括`conv2d`、`depthwise_conv2d`和`mul`等。量化训练会对所有的`conv2d`、`depthwise_conv2d`和`mul`进行量化操作，且要求它们的输入中必须包括激活和参数两部分。

#### 2.1.1 基于High-Level API的量化训练

>注意：多个压缩策略组合使用时，量化训练策略必须放在最后。

```
class Compressor(object):
    def __init__(self,
                 place,
                 scope,
                 train_program,
                 train_reader=None,
                 train_feed_list=None,
                 train_fetch_list=None,
                 eval_program=None,
                 eval_reader=None,
                 eval_feed_list=None,
                 eval_fetch_list=None,
                 teacher_programs=[],
                 checkpoint_path='./checkpoints',
                 train_optimizer=None,
                 distiller_optimizer=None):
```
在定义Compressor对象时，需要注意以下问题：

- train program如果带反向operators和优化更新相关的operators, train_optimizer需要设置为None.
- eval_program中parameter的名称需要与train_program中的parameter的名称完全一致。
- 最终保存的量化后的int8模型，是在eval_program网络基础上进行剪枝保存的，所以，如果用户希望最终保存的模型可以用于inference, 则eval program需要包含infer需要的各种operators.
- checkpoint保存的是float数据类型的模型

在配置文件中，配置量化训练策略发方法如下：
```
strategies:
    quantization_strategy:
        class: 'QuantizationStrategy'
        start_epoch: 0
        end_epoch: 10
        float_model_save_path: './output/float'
        mobile_model_save_path: './output/mobile'
        int8_model_save_path: './output/int8'
        weight_bits: 8
        activation_bits: 8
        weight_quantize_type: 'abs_max'
        activation_quantize_type: 'abs_max'
        save_in_nodes: ['image']
        save_out_nodes: ['quan.tmp_2']
 compressor:
    epoch: 20
    checkpoint_path: './checkpoints_quan/'
    strategies:
        - quantization_strategy
```
可配置参数有：

- **class:** 量化策略的类名称，目前仅支持`QuantizationStrategy`
- **start_epoch:** 在start_epoch开始之前，量化训练策略会往train_program和eval_program插入量化operators和反量化operators. 从start_epoch开始，进入量化训练阶段。
- **end_epoch:** 在end_epoch结束之后，会保存用户指定格式的模型。注意：end_epoch之后并不会停止量化训练，而是继续训练到compressor.epoch为止。
- **float_model_save_path:**  保存float数据格式模型的路径。模型weight的实际大小在int8可表示范围内，但是是以float格式存储的。如果设置为None, 则不存储float格式的模型。默认为None.
- **int8_model_save_path:** 保存int8数据格式模型的路径。如果设置为None, 则不存储int8格式的模型。默认为None.
- **mobile_model_save_path:** 保存兼容paddle-mobile框架的模型的路径。如果设置为None, 则不存储mobile格式的模型。默认为None.
- **weight_bits:** 量化weight的bit数，bias不会被量化。
- **activation_bits:** 量化activation的bit数。
-  **weight_quantize_type:** 对于weight的量化方式，目前支持'abs_max'， 'channel_wise_abs_max'.
- **activation_quantize_type:** 对activation的量化方法，目前可选`abs_max`或`range_abs_max`。`abs_max`意为在训练的每个step和inference阶段动态的计算量化范围。`range_abs_max`意为在训练阶段计算出一个静态的范围，并将其用于inference阶段。
- **save_in_nodes:** variable名称列表。在保存量化后模型的时候，需要根据save_in_nodes对eval programg 网络进行前向遍历剪枝。默认为eval_feed_list内指定的variable的名称列表。
- **save_out_nodes:** varibale名称列表。在保存量化后模型的时候，需要根据save_out_nodes对eval programg 网络进行回溯剪枝。默认为eval_fetch_list内指定的variable的名称列表。


#### 2.1.2 基于Low-Level API的量化训练

量化训练High-Level API是对Low-Level API的高层次封装，这使得用户仅需编写少量的代码和配置文件即可进行量化训练。然而，封装必然会带来使用灵活性的降低。因此，若用户在进行量化训练时需要更多的灵活性，可参考 [量化训练Low-Level API使用示例](../quant_low_level_api/README.md) 。

### 2.2 卷积核剪切
该策略通过减少指定卷积层中卷积核的数量，达到缩减模型大小和计算复杂度的目的。根据选取剪切比例的策略的不同，又细分为以下两个方式：

- uniform pruning: 每层剪切掉相同比例的卷积核数量。
- sensitive pruning: 根据每层敏感度，剪切掉不同比例的卷积核数量。

两种剪切方式都需要加载预训练模型。
卷积核剪切是基于结构剪切，所以在配置文件中需要注册一个`StructurePruner`,  如下所示：

```
pruners:
    pruner_1:
        class: 'StructurePruner'
        pruning_axis:
            '*': 0
        criterions:
            '*': 'l1_norm'
```

其中，一个配置文件可注册多个pruners, 所有pruner需要放在`pruners`关键字下, `pruner`的可配置参数有：

- **class:** pruner 的类型，目前只支持`StructurePruner`
- **pruning_axis:** 剪切的纬度；'`conv*': 0`表示对所有的卷积层filter weight的第0维进行剪切，即对卷积层filter的数量进行剪切。
- **criterions**： 通过通配符指定剪切不同parameter时用的排序方式。目前仅支持`l1_norm`.


#### 2.2.1 uniform pruning

uniform pruning剪切策略需要在配置文件的`strategies`关键字下注册`UniformPruneStrategy`实例，并将其添加至compressor的strategies列表中。
如下所示：
```
strategies:
    uniform_pruning_strategy:
        class: 'UniformPruneStrategy'
        pruner: 'pruner_1'
        start_epoch: 0
        target_ratio: 0.5
        pruned_params: '.*_sep_weights'
compressor:
    epoch: 100
    strategies:
        - uniform_pruning_strategy
```
UniformPruneStrategy的可配置参数有：

- **class:** 如果使用Uniform剪切策略，请设置为`UniformPruneStrategy`
- **pruner:** StructurePruner实例的名称，需要在配置文件中注册。在pruner中指定了对单个parameter的剪切方式。
- **start_epoch:** 开始剪切策略的epoch. 在start_epoch开始之前，该策略会对网络中的filter数量进行剪切，从start_epoch开始对被剪切的网络进行fine-tune训练，直到整个压缩任务结束。
- **target_ratio:** 将目标网络的FLOPS剪掉的比例。
- **pruned_params:** 被剪切的parameter的名称，支持通配符。如，‘*’为对所有parameter进行剪切，‘conv*’意为对所有名义以‘conv’开头的parameter进行剪切。



#### 2.2.2  sensitive pruning

sensitive剪切策略需要在配置文件的`strategies`关键字下注册`SensitivePruneStrategy`实例，并将其添加至compressor的strategies列表中。
如下所示：
```
strategies:
    sensitive_pruning_strategy:
        class: 'SensitivePruneStrategy'
        pruner: 'pruner_1'
        start_epoch: 0
        delta_rate: 0.1
        target_ratio: 0.5
        num_steps: 1
        eval_rate: 0.2
        pruned_params: '.*_sep_weights'
        sensitivities_file: 'mobilenet_acc_top1_sensitive.data'
        metric_name: 'acc_top1'
compressor:
    epoch: 200
    strategies:
        - sensitive_pruning_strategy
```
SensitivePruneStrategy可配置的参数有：

- **class:** 如果使用敏感度剪切策略，请设置为`SensitivePruneStrategy`
- **pruner:** StructurePruner实例的名称，需要在配置文件中注册。在pruner中指定了对单个parameter的剪切方式。
- **start_epoch:** 开始剪切策略的epoch。 在start_epoch开始之前，该策略会对网络中的filter数量进行第一次剪切。
- **delta_rate:** 统计敏感度信息时，剪切率从0到1，依次递增delta_rate. 具体细节可参考[原理介绍文档]()
- **target_ratio:** 将目标网络的FLOPS剪掉的比例。
- **num_steps:** 整个剪切过程的步数。每次迭代剪掉的比例为：$step = 1 - (1-target\_ratio)^{\frac{1}{num\_steps}}$
- **eval_rate:** 计算敏感度时，随机抽取使用的验证数据的比例。在迭代剪切中，为了快速重新计算每一步的每个parameter的敏感度，建议随机选取部分验证数据进行计算。当`num_steps`等于1时，建议使用全量数据进行计算。


### 2.3 蒸馏

PaddleSlim支持`FSP_loss`, `L2_loss`和`softmax_with_cross_entropy_loss`, 用户可以在配置文件中，用这三种loss组合teacher net和student net的任意一层。

与其它策略不同，如果要使用蒸馏策略，用户在脚本中构造Compressor对象时，需要指定teacher program 和distiller optimizer.
其中，teacher program有以下要求：

- teacher program需要加载预训练好的模型。
- teacher program中的变量不能与student program中的变量有命名冲突。
- teacher program中只有前向计算operators， 不能有backward operators。
- 用户不必手动设置teacher program的stop_gradient属性(不计算gradient和不更新weight)，PaddleSlim会自动将其设置为True.

distiller optimizer用来为student net和teacher net组合而成的网络添加反向operators和优化相关的operators, 仅用于蒸馏训练阶段。

在配置文件中，配置蒸馏策略方式如下：
```
strategies:
    distillation_strategy:
        class: 'DistillationStrategy'
        distillers: ['fsp_distiller', 'l2_distiller']
        start_epoch: 0
        end_epoch: 130
```
其中， 需要在关键字`strategies`下注册策略实例，可配置参数有：

- **class:** 策略类的名称，蒸馏策略请设置为DistillationStrategy。
- **distillers:** 一个distiller列表，列表中每一个distiller代表了student net和teacher net之间的一个组合loss。该策略会将这个列表中定义的loss加在一起作为蒸馏训练阶段优化的目标。 distiller需要提前在当前配置文件中进行注册，下文会详细介绍其注册方法。
- **start_epoch:** 在start_epoch开始之前，该策略会根据用户定义的losses将teacher net合并到student net中，并根据合并后的loss添加反向计算操作和优化更新操作。
- **end_epoch:** 在 end_epoch结束之后，该策略去将teacher net从student net中删除，并回复student net的loss. 在次之后，进入单独fine-tune student net的阶段。

distiller的配置方式如下：

**FSPDistiller**
```
distillers:
    fsp_distiller:
        class: 'FSPDistiller'
        teacher_pairs: [['res2a_branch2a.conv2d.output.1.tmp_0', 'res3a_branch2a.conv2d.output.1.tmp_0']]
        student_pairs: [['depthwise_conv2d_1.tmp_0', 'conv2d_3.tmp_0']]
        distillation_loss_weight: 1
```
- **class:** distiller类名称，可选：`FSPDistiller`，`L2Distiller`，`SoftLabelDistiller`
- **teacher_pairs:**  teacher网络对应的sections. 列表中每一个section由两个variable name表示，这两个variable代表网络中的两个feature map. 这两个feature map可以有不同的channel数量，但是必须有相同的长和宽。
- **student_pairs:** student网络对应的sections. student_pairs[i]与teacher_pairs[i]计算出一个fsp loss.
- **distillation_loss_weight:** 当前定义的fsp loss对应的权重。默认为1.0

**L2-loss**

```
distillers:
        l2_distiller:
                class: 'L2Distiller'
                teacher_feature_map: 'fc_1.tmp_0'
                student_feature_map: 'fc_0.tmp_0'
                distillation_loss_weight: 1
```

- **teacher_feature_map:** teacher网络中用于计算l2 loss的feature map
- **student_feature_map:** student网络中用于计算l2 loss的feature map， shape必须与`teacher_feature_map`完全一致。

**SoftLabelDistiller**

```
distillers:
    soft_label_distiller:
        class: 'SoftLabelDistiller'
        student_temperature: 1.0
        teacher_temperature: 1.0
        teacher_feature_map: 'teacher.tmp_1'
        student_feature_map: 'student.tmp_1'
        distillation_loss_weight: 0.001
```

- **teacher_feature_map:** teacher网络中用于计算softmax_with_cross_entropy的feature map。
- **student_feature_map:** student网络中用于计算softmax_with_cross_entropy的feature map。shape必须与`teacher_feature_map`完全一致。
- **student_temperature:** 在计算softmax_with_cross_entropy之前，用该系数除student_feature_map。
- **teacher_temperature:** 在计算softmax_with_cross_entropy之前，用该系数除teacher_feature_map。
- **distillation_loss_weight:** 当前定义的loss对应的权重。默认为1.0
