# 离线量化

本文首先简要概述量化的原理，然后详细说明离线量化的使用方法，最后给出MobileNetV1离线量化示例。如果想快速上手使用，可以跳过原理部分，直接阅读使用方法和示例部分。

## 离线量化原理概述

### 模型量化
模型量化是使用更少的比特数（如8-bit、3-bit、2-bit等）表示神经网络的权重和激活。模型量化可以加快推理速度、减小存储大小、降低功耗等优点。目前，模型量化主要分为离线量化（Post Training Quantization）和QAT量化（Quantization Aware Training）。

### 量化方式
将FP32类型Tensor转换为INT8类型Tensor的过程相当于信息再编码（re-encoding information ），且要求再编码后精度损失要尽量小。FP32和INT8类型Tensor可以通过如下线性映射实现相互转换：
$$ f = s * q + b $$
$$ q = round((f-b)/s)$$
其中，`f`为FP32类型的Tensor，`s`为量化比例因子，`q`为INT8类型的Tensor，`b`为FP32类型的Bias，`round`是取整。

如果使用对称量化，即将FP32的数值量化到`-127~127`范围内，则不再需要偏置：
$$ f  = s * q $$
$$ q  = round(f / s) $$

### 离线量化
离线量化是基于采样数据，采用KL散度等方法计算量化比例因子的方法。相比QAT量化，离线量化不需要重新训练，可以快速得到量化模型。

离线量化的目标是求取量化比例因子，主要有两种方法：非饱和量化方法 ( No Saturation) 和饱和量化方法 (Saturation)。非饱和量化方法计算FP32类型Tensor中绝对值的最大值`abs_max`，将其映射为127，则量化比例因子等于`abs_max/127`。饱和量化方法使用KL散度计算一个合适的阈值`T` (`T<mab_max`)，将其映射为127，则量化比例因子等于`T/127`。一般而言，对于待量化op的权重Tensor，采用非饱和量化方法，对于待量化op的激活Tensor（包括输入和输出），采用饱和量化方法 。

离线量化的内部实现步骤：
* 加载预训练的FP32模型，配置`DataLoader`；
* 读取样本数据，执行模型的前向推理，保存待量化op的激活Tensor的数值；
* 基于激活Tensor的采样数据，使用饱和量化方法计算它的量化比例因子；
* 模型权重Tensor数据一直保持不变，使用非饱和方法计算它每个通道的绝对值最大值，作为每个通道的量化比例因子；
* 将FP32模型转成INT8模型，进行保存。

## 离线量化使用说明

1）**准备模型和样本数据**

首先，需要准备已经训练好的FP32预测模型，即 `save_inference_model()` 保存的模型。离线量化读取样本数据进行前向计算，所以需要准备样本数据。样本数据最好是预测数据的中具有代表性的一部分，这样可以计算得到更加准确的量化比例因子。样本数据的数量可以是100~500，当然样本数据越多，计算的的量化比例因子越准确。

2）**配置读取样本数据的接口**

离线量化内部使用异步数据读取的方式读取样本，用户只需要根据模型的输入，配置读取样本数据的sample_generator。sample_generator是Python生成器，用作`DataLoader.set_sample_generator()`的数据源，**必须每次返回单个样本**。建议参考官方文档[异步数据读取](https://www.paddlepaddle.org.cn/documentation/docs/zh/user_guides/howto/prepare_data/use_py_reader.html)。

3）**调用离线量化**

机器上安装PaddlePaddle，然后调用PostTrainingQuantization实现离线量化，以下对api接口进行详细介绍。

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
                 is_full_quantize=False)
```
调用上述api，传入离线量化必要的参数。参数说明：
* executor：执行模型的executor，可以在cpu或者gpu上执行。
* sample_generator：第二步中配置的读取样本数据的接口。
* model_dir：待量化模型的路径，其中保存模型文件和权重文件。
* model_filename：待量化模型的模型文件名，如果模型文件名不是`__model__`，则需要自行设置模型文件名。
* params_filename：待量化模型的权重文件名，如果所有权重保存成一个文件，则需要自行设置权重文件名。
* batch_size：一次读取样本数据的数量，可以随意设置。
* batch_nums：读取样本数据的次数。如果设置为None，则从sample_generator中读取所有样本数据进行离线量化；如果设置为非None，则从sample_generator中读取`batch_size*batch_nums`个样本数据。
* scope：模型scope，默认为None，则会使用global_scope()。
* algo：计算待量化激活Tensor的量化比例因子的方法。设置为KL，则使用KL散度方法，设置为direct，则使用abs max方法。
* quantizable_op_type: 需要量化的op类型，默认是`["conv2d", "depthwise_conv2d", "mul"]`，列表中的值可以是任意支持量化的op类型。
* is_full_quantize：是否进行全量化。设置为True，则对模型中所有支持量化的op进行量化；设置为False，则只对`quantizable_op_type` 中op类型进行量化。

```
PostTrainingQuantization.quantize()
```
调用上述接口开始离线量化。根据样本数量、模型的大小和量化op类型不同，离线量化需要的时间也不一样。比如使用100图片对`MobileNetV1`进行离线量化，花费大概1分钟。

```
PostTrainingQuantization.save_quantized_model(save_model_path)
```
调用上述接口保存离线量化模型，其中save_model_path为保存的路径。


## 离线量化使用示例

下面以MobileNetV1为例，介绍离线量化Low-Level API的使用方法。

> 该示例的代码放在[models/PaddleSlim/quant_low_level_api/](https://github.com/PaddlePaddle/models/tree/develop/PaddleSlim/quant_low_level_api)目录下。如果需要执行该示例，首先clone下来[models](https://github.com/PaddlePaddle/models.git)，然后执行[run_post_training_quanzation.sh](run_post_training_quanzation.sh)脚本，最后量化模型保存在`mobilenetv1_int8_model`。

1） 准备工作

安装PaddlePaddle，准备已经训练好的FP32预测模型。

准备样本数据，文件结构如下。val文件夹中有100张图片，val_list.txt文件中包含图片的label。如果特定输入不会影响模型的前向计算，则该输入的数据可以随意设置。
```bash
samples_100
└──val
└──val_list.txt
```

2）配置读取样本数据的接口

MobileNetV1的输入是图片和标签，所以配置读取样本数据的sample_generator，每次返回一张图片和一个标签。详细代码在[models/PaddleSlim/reader.py](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/reader.py)。

3）调用离线量化

调用离线量化的核心代码如下，详细代码在[post_training_quantization.py](post_training_quantization.py)。
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
