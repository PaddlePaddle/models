# Linux GPU/CPU PACT量化训练功能开发文档

# 目录

- [1. 简介](#1)
- [2. 量化训练功能开发](#2)
    - [2.1 准备数据和环境](#2.1)
    - [2.2 准备待量化模型](#2.2)
    - [2.3 开始量化训练及保存模型](#2.3)
    - [2.4 验证推理结果正确性](#2.4)
- [3. FAQ](#3)
    - [3.1 通用问题](#3.1)


<a name="1"></a>

## 1. 简介

Paddle 量化训练（Quant-aware Training, QAT）是指在训练过程中对模型的权重及激活做模拟量化，并且产出量化训练校准后的量化模型，使用该量化模型进行预测，可以减少计算量、降低计算内存、减小模型大小。

更多关于PaddleSlim 量化的介绍，可以参考[PaddleSlim 量化训练官网教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/dygraph/quanter/qat.rst#%E9%87%8F%E5%8C%96%E8%AE%AD%E7%BB%83)。

<a name="2"></a>

## 2. 量化训练功能开发

Paddle 混合精度训练开发可以分为4个步骤，如下图所示。

<div align="center">
    <img src="../images/quant_aware_training_guide.png" width="600">
</div>

其中设置了2个核验点，分别为：

* 准备待量化模型
* 验证量化模型推理结果正确性

<a name="2.1"></a>

### 2.1 准备数据和环境

**【准备校准数据】**

将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

选择适量训练集或验证集

**【准备开发环境】**

- 确定已安装paddle，通过pip安装linux版本paddle命令如下，更多的版本安装方法可查看飞桨[官网](https://www.paddlepaddle.org.cn/)
- 确定已安装paddleslim，通过pip安装linux版本paddle命令如下，更多的版本安装方法可查看[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)

```
pip install paddlepaddle-gpu==2.2.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install paddleslim==2.2.1
```

<a name="2.2"></a>

### 2.2 准备待量化模型

**【基本流程】**

- Step1：定义继承自`paddle.nn.Layer`的网络模型

**【实战】**

模型组网可以参考[mobilenet_v3](https://github.com/PaddlePaddle/models/blob/release/2.2/tutorials/mobilenetv3_prod/Step6/paddlevision/models/mobilenet_v3.py)

```python
fp32_model = mobilenet_v3_small()
```

<a name="2.3"></a>

### 2.3 开始量化训练及保存模型

**【基本流程】**

使用飞桨PaddleSlim中的`QAT`接口开始进行量化训练：

- Step1：配置量化训练参数。

```python
quant_config = {
    'weight_preprocess_type': None,
    'activation_preprocess_type': None,
    'weight_quantize_type': 'channel_wise_abs_max',
    'activation_quantize_type': 'moving_average_abs_max',
    'weight_bits': 8,
    'activation_bits': 8,
    'dtype': 'int8',
    'window_size': 10000,
    'moving_rate': 0.9,
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}
```

- `activation_preprocess_type`'：代表对量化模型激活值预处理的方法，目前支持PACT方法，如需使用可以改为'PACT'；默认为None，代表不对激活值进行任何预处理。
- `weight_preprocess_type`：代表对量化模型权重参数预处理的方法；默认为None，代表不对权重进行任何预处理。
- `weight_quantize_type`：代表模型权重的量化方式，可选的有['abs_max', 'moving_average_abs_max', 'channel_wise_abs_max']，默认为channel_wise_abs_max
- `activation_quantize_type`：代表模型激活值的量化方式，可选的有['abs_max', 'moving_average_abs_max']，默认为moving_average_abs_max
- `quantizable_layer_type`：代表量化OP的类型，目前支持Conv2D和Linear


- Step2：插入量化算子，得到量化训练模型

```python
from paddleslim.dygraph.quant import QAT
quanter = QAT(config=quant_config)
quanter.quantize(net)
```

- Step3：开始训练。

- Step4：量化训练结束，保存量化模型

```python
quanter.save_quantized_model(net, 'save_dir', input_spec=[paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32')])
```


**【实战】**

量化训练配置、训练及保存量化模型请参考[MobileNetv3量化训练文档](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6/docs/train_pact_infer_python.md)

<a name="2.4"></a>

### 2.4 验证推理结果正确性

**【基本流程】**

使用Paddle Inference库测试离线量化模型，确保模型精度符合预期。

- Step1：初始化`paddle.inference`库并配置相应参数

```python
import paddle.inference as paddle_infer
model_file = os.path.join('quant_model', 'qat_inference.pdmodel')
params_file = os.path.join('quant_model', 'qat_inference.pdiparams')
config = paddle_infer.Config(model_file, params_file)
if FLAGS.use_gpu:
    config.enable_use_gpu(1000, 0)
if not FLAGS.ir_optim:
    config.switch_ir_optim(False)

predictor = paddle_infer.create_predictor(config)
```

- Step2：配置预测库输入输出

```python
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
```

- Step3：开始预测并检验结果正确性

```python
input_handle.copy_from_cpu(img_np)
predictor.run()
 output_data = output_handle.copy_to_cpu()
```

**【实战】**


1）初始化`paddle.inference`库并配置相应参数：

具体可以参考MobileNetv3 [Inference模型测试代码](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6/deploy/ptq_python/eval.py)

2）配置预测库输入输出：

具体可以参考MobileNetv3 [Inference模型测试代码](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6/deploy/ptq_python/eval.py)

3）开始预测：

具体可以参考MobileNetv3 [Inference模型测试代码](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6/deploy/ptq_python/eval.py)

4）测试单张图像预测结果是否正确，可参考[Inference预测文档](https://github.com/PaddlePaddle/models/blob/release/2.2/docs/tipc/train_infer_python/infer_python.md)

5）同时也可以测试量化模型和FP32模型的精度，确保量化后模型精度损失符合预期。参考[MobileNet量化模型精度验证文档](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6/deploy/ptq_python/README.md)

<a name="3"></a>

## 3. FAQ

### 3.1 通用问题
如果您在使用该文档完成PACT量化训练的过程中遇到问题，可以给在[这里](https://github.com/PaddlePaddle/PaddleSlim/issues)提一个ISSUE，我们会高优跟进。
