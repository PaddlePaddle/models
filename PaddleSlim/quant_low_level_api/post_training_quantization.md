<div align="center">
  <h3>
    <a href="./README.md">
      模型量化概述
    </a>
    <span> | </span>
    <a href="../docs/tutorial.md">
      模型量化原理
    </a>
    <span> | </span>
    <a href="./quantization_aware_training.md">
      量化训练使用方法和示例
    </a>
    <span> | </span>
    <a href="./post_training_quantization.md">
      训练后量化使用方法和示例
    </a>
  </h3>
</div>

---
# 训练后量化Low-Level API使用方法和示例

## 目录

- [训练后量化使用说明](#1-训练后量化使用说明)
- [训练后量化使用示例](#2-训练后量化使用示例)

## 1. 训练后量化使用说明

1）**准备模型和校准数据**

首先，需要准备已经训练好的FP32预测模型，即 `save_inference_model()` 保存的模型。训练后量化读取校准数据进行前向计算，所以需要准备校准数据集。校准数据集应为测试集（或训练集）中具有代表性的一部分，如随机取出的部分数据，这样可以计算得到更加准确的量化比例因子。建议样本数据的数量为100~500。

2）**配置校准数据生成器**

训练后量化内部使用异步数据读取的方式读取校准数据，用户只需要根据模型的输入，配置读取数据的sample_generator。sample_generator是Python生成器，用作`DataLoader.set_sample_generator()`的数据源，**必须每次返回单个样本**。建议参考官方文档[异步数据读取](https://www.paddlepaddle.org.cn/documentation/docs/zh/user_guides/howto/prepare_data/use_py_reader.html)。

3）**调用训练后量化**

机器上安装PaddlePaddle develop分支编译的whl包，然后调用PostTrainingQuantization实现训练后量化，以下对api接口进行详细介绍。

``` python
class PostTrainingQuantization(
    executor,
    sample_generator,
    model_dir,
    model_filename=None,
    params_filename=None,
    batch_size=10,
    batch_nums=None,
    scope=None,
    algo="KL",
    quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
    is_full_quantize=False,
    is_use_cache_file=False,
    cache_dir="./temp_post_training")
```
调用上述api，传入训练后量化必要的参数。参数说明：
* executor：执行模型的executor，可以在cpu或者gpu上执行。
* sample_generator：第二步中配置的校准数据生成器。
* model_dir：待量化模型的路径，其中保存模型文件和权重文件。
* model_filename：待量化模型的模型文件名，如果模型文件名不是`__model__`，则需要使用model_filename设置模型文件名。
* params_filename：待量化模型的权重文件名，如果所有权重保存成一个文件，则需要使用params_filename设置权重文件名。
* batch_size：一次读取样本数据的数量。
* batch_nums：读取样本数据的次数。如果设置为None，则从sample_generator中读取所有样本数据进行训练后量化；如果设置为非None，则从sample_generator中读取`batch_size*batch_nums`个样本数据。
* scope：模型运行时使用的scope，默认为None，则会使用global_scope()。
* algo：计算待量化激活Tensor的量化比例因子的方法。设置为`KL`，则使用KL散度方法，设置为`direct`，则使用abs max方法。默认为`KL`。
* quantizable_op_type: 需要量化的op类型，默认是`["conv2d", "depthwise_conv2d", "mul"]`，列表中的值可以是任意支持量化的op类型。
* is_full_quantize：是否进行全量化。设置为True，则对模型中所有支持量化的op进行量化；设置为False，则只对`quantizable_op_type` 中op类型进行量化。目前，支持的量化类型如下：'conv2d', 'depthwise_conv2d', 'mul', "pool2d", "elementwise_add", "concat", "softmax", "argmax", "transpose", "equal", "gather", "greater_equal", "greater_than", "less_equal", "less_than", "mean", "not_equal", "reshape", "reshape2", "bilinear_interp", "nearest_interp", "trilinear_interp", "slice", "squeeze", "elementwise_sub"。
* is_use_cache_file：是否使用缓存文件。如果设置为True，训练后量化过程中的采样数据会保存到磁盘文件中；如果设置为False，所有采样数据会保存到内存中。当待量化的模型很大或者校准数据数量很大，建议设置is_use_cache_file为True。默认为False。
* cache_dir：当is_use_cache_file等于True，会将采样数据保存到该文件中。量化完成后，该文件中的临时文件会自动删除。

```
PostTrainingQuantization.quantize()
```
调用上述接口开始训练后量化。根据样本数量、模型的大小和量化op类型不同，训练后量化需要的时间也不一样。比如使用ImageNet2012数据集中100图片对`MobileNetV1`进行训练后量化，花费大概1分钟。

```
PostTrainingQuantization.save_quantized_model(save_model_path)
```
调用上述接口保存训练后量化模型，其中save_model_path为保存的路径。

**训练后量化支持部分量化功能**
* 方法1：设置quantizable_op_type，则只会对quantizable_op_type中的Op类型进行量化，模型中其他Op类型保持不量化。
* 方法2：构建网络的时候，将不需要量化的特定Op定义在 `skip_quant` 的name_scope中，则可以跳过特定Op的量化，示例如下。
```python
with fluid.name_scope('skip_quant'):
    pool = fluid.layers.pool2d(input=hidden, pool_size=2, pool_type='avg', pool_stride=2)
    # 不对pool2d进行量化
```

## 2. 训练后量化使用示例

下面以MobileNetV1为例，介绍训练后量化Low-Level API的使用方法。

> 该示例的代码放在[models/PaddleSlim/quant_low_level_api/](https://github.com/PaddlePaddle/models/tree/develop/PaddleSlim/quant_low_level_api)目录下。如果需要执行该示例，首先clone下来[models](https://github.com/PaddlePaddle/models.git)，然后执行[run_post_training_quanzation.sh](run_post_training_quanzation.sh)脚本，最后量化模型保存在`mobilenetv1_int8_model`目录下。

1）**准备模型和校准数据**

安装最新版PaddlePaddle，准备已经训练好的FP32预测模型。

准备校准数据，文件结构如下。val文件夹中有100张图片，val_list.txt文件中包含图片的label。
```bash
samples_100
└──val
└──val_list.txt
```

2）**配置校准数据生成器**

MobileNetV1的输入是图片和标签，所以配置读取校准数据的sample_generator，每次返回一张图片和一个标签。详细代码在[models/PaddleSlim/reader.py](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/reader.py)。

3）**调用训练后量化**

调用训练后量化的核心代码如下，详细代码在[post_training_quantization.py](post_training_quantization.py)。
``` python
place = fluid.CUDAPlace(0) if args.use_gpu == "True" else fluid.CPUPlace()
exe = fluid.Executor(place)
sample_generator = reader.val(data_dir=args.data_path)

ptq = PostTrainingQuantization(
    executor=exe,
    sample_generator=sample_generator,
    model_dir=args.model_dir,
    model_filename=args.model_filename,
    params_filename=args.params_filename,
    batch_size=args.batch_size,
    batch_nums=args.batch_nums,
    algo=args.algo,
    is_full_quantize=args.is_full_quantize == "True")
quantized_program = ptq.quantize()
ptq.save_quantized_model(args.save_model_path)
```
4）**测试训练后量化模型精度**

使用ImageNet2012测试集中100张图片做校准数据集，对`conv2d`, `depthwise_conv2d`, `mul`, `pool2d`, `elementwise_add`和`concat`进行训练后量化，然后在ImageNet2012验证集上测试。下表列出了常见分类模型训练后量化前后的精度对比。

模型 | FP32 Top1 | FP32 Top5 | INT8 Top1 | INT8 Top5| Top1 Diff | Tp5 Diff
-|:-:|:-:|:-:|:-:|:-:|:-:
googlenet   | 70.50% | 89.59% | 70.12% | 89.38% | -0.38% | -0.21%
mobilenetv1 | 70.91% | 89.54% | 70.24% | 89.03% | -0.67% | -0.51%
mobilenetv2 | 71.90% | 90.56% | 71.36% | 90.17% | -0.54% | -0.39%
resnet50    | 76.35% | 92.80% | 76.26% | 92.81% | -0.09% | +0.01%
resnet101   | 77.49% | 93.57% | 75.44% | 92.56% | -2.05% | -1.01%
vgg16       | 72.08% | 90.63% | 71.93% | 90.64% | -0.15% | +0.01%
vgg19       | 72.56% | 90.83% | 72.55% | 90.77% | -0.01% | -0.06%
