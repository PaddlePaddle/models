# MobileNetV3

## 目录


- [1. 简介](#1)
- [2. 数据集和复现精度](#2)
- [3. 准备数据与环境](#3)
    - [3.1 准备环境](#3.1)
    - [3.2 准备数据](#3.2)
    - [3.3 准备模型](#3.3)
- [4. 开始使用](#4)
    - [4.1 模型训练](#4.1)
    - [4.2 模型评估](#4.2)
    - [4.3 模型预测](#4.3)
- [5. 模型推理部署](#5)
- [6. TIPC自动化测试脚本](#6)
- [7. 参考链接与文献](#7)

<a name="1"></a>

## 1. 简介

MobileNetV3 是 2019 年提出的一种基于 NAS 的新的轻量级网络，为了进一步提升效果，将 relu 和 sigmoid 激活函数分别替换为 hard_swish 与 hard_sigmoid 激活函数，同时引入了一些专门减小网络计算量的改进策略，最终性能超越了当时其他的轻量级骨干网络。



**论文:** [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

**参考repo:** [https://github.com/pytorch/vision](https://github.com/pytorch/vision)


在此感谢[vision](https://github.com/pytorch/vision)，提高了MobileNetV3论文复现的效率。

注意：在这里为了简化流程，仅关于`ImageNet标准训练过程`做训练对齐，具体地：
* 训练总共120epoch，总的batch size是256*8=2048，学习率为0.8，下降策略为Piecewise Decay(30epoch下降10倍)
* 训练预处理：RandomResizedCrop(size=224) + RandomFlip(p=0.5) + Normalize
* 评估预处理：Resize(256) + CenterCrop(224) + Normalize

这里`mobilenet_v3_small`的参考指标也是重新训练得到的。

<a name="2"></a>

## 2. 数据集和复现精度

数据集为ImageNet，训练集包含1281167张图像，验证集包含50000张图像。

您可以从[ImageNet 官网](https://image-net.org/)申请下载数据。


| 模型      | top1/5 acc (参考精度) | top1/5 acc (复现精度) | 下载链接 |
|:---------:|:------:|:----------:|:----------:|
| Mo | -/-   | 0.601/0.826   | [预训练模型](https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams) \|  [Inference模型(coming soon!)]() \| [日志](https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/train_mobilenet_v3_small.log) |

<a name="3"></a>

## 3. 准备环境与数据


<a name="3.1"></a>

### 3.1 准备环境

* 下载代码

```bash
git clone https://github.com/PaddlePaddle/models.git
cd models/tutorials/mobilenetv3_prod/Step6
```

* 安装paddlepaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。

```bash
# 需要安装2.2及以上版本的Paddle，如果
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0
# 安装CPU版本的Paddle
pip install paddlepaddle==2.2.0
```

安装完成之后，可以使用下面的命令验证是否安装成功

```python
import paddle
paddle.utils.run_check()
```

如果出现了`PaddlePaddle is installed successfully!`等输出内容，如下所示，则说明安装成功。

```
W1223 02:51:03.061575 33723 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 10.2
W1223 02:51:03.070878 33723 device_context.cc:465] device: 0, cuDNN Version: 7.6.
PaddlePaddle works well on 1 GPU.
W1223 02:51:33.185979 33723 fuse_all_reduce_op_pass.cc:76] Find all_reduce operators: 2. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 2.
PaddlePaddle works well on 8 GPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 安装requirements

```bash
pip install -r requirements.txt
```

<a name="3.2"></a>

### 3.2 准备数据

如果您已经下载好ImageNet1k数据集，那么该步骤可以跳过，如果您没有，则可以从[ImageNet官网](https://image-net.org/download.php)申请下载。

如果只是希望快速体验模型训练功能，则可以直接解压`test_images/lite_data.tar`，其中包含16张训练图像以及16张验证图像。

```bash
tar -xf test_images/lite_data.tar
```

执行该命令后，会在当前路径下解压出对应的数据集文件夹lite_data


<a name="3.3"></a>

### 3.3 准备模型

如果您希望直接体验评估或者预测推理过程，可以直接根据[第2节：数据集和复现精度]()的内容下载提供的预训练模型，直接体验模型评估、预测、推理部署等内容。

使用下面的命令下载模型

```bash
# 下载预训练模型
wget https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams
# 下载推理模型
# coming soon!
```


<a name="4"></a>

## 4. 开始使用

<a name="4.1"></a>

### 4.1 模型训练

* 单机单卡训练

```bash
export CUDA_VISIBLE_DEVICES=0
python3 train.py --data-path=./ILSVRC2012 --lr=0.1 --batch-size=256
```

部分训练日志如下所示。

```
[Epoch 1, iter: 4780] top1: 0.10312, top5: 0.27344, lr: 0.01000, loss: 5.34719, avg_reader_cost: 0.03644 sec, avg_batch_cost: 0.05536 sec, avg_samples: 64.0, avg_ips: 1156.08863 images/sec.
[Epoch 1, iter: 4790] top1: 0.08750, top5: 0.24531, lr: 0.01000, loss: 5.28853, avg_reader_cost: 0.05164 sec, avg_batch_cost: 0.06852 sec, avg_samples: 64.0, avg_ips: 934.08427 images/sec.
```

* 单机多卡训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch --gpus="0,1,2,3" train.py --data-path="./ILSVRC2012" --lr=0.4 --batch-size=256
```

更多配置参数可以参考[train.py](./train.py)的`get_args_parser`函数。

<a name="4.2"></a>

### 4.2 模型评估

该项目中，训练与评估脚本相同，指定`--test-only`参数即可完成预测过程。

```bash
python train.py --test-only --data-path=/paddle/data/ILSVRC2012 --pretrained=./mobilenet_v3_small_paddle.pdparams
```

期望输出如下。

```
Test:  [   0/1563]  eta: 1:14:20  loss: 1.0456 (1.0456)  acc1: 0.7812 (0.7812)  acc5: 0.9062 (0.9062)  time: 2.8539  data: 2.8262
...
Test:  [1500/1563]  eta: 0:00:05  loss: 1.2878 (1.9196)  acc1: 0.7344 (0.5639)  acc5: 0.8750 (0.7893)  time: 0.0623  data: 0.0534
Test: Total time: 0:02:05
 * Acc@1 0.564 Acc@5 0.790
```

<a name="4.3"></a>

### 4.3 模型预测

* 使用GPU预测

```
python tools/predict.py --pretrained=./mobilenet_v3_small_pretrained.pdparams --img-path=images/demo.jpg
```

对于下面的图像进行预测

<div align="center">
    <img src="./images/demo.jpg" width=300">
</div>

最终输出结果为`class_id: 8, prob: 0.9091238975524902`，表示预测的类别ID是`8`，置信度为`0.909`。

* 使用CPU预测

```
python tools/predict.py --pretrained=./mobilenet_v3_small_pretrained.pdparams --img-path=images/demo.jpg --device=cpu
```

<a name="5"></a>

## 5. 模型推理部署

### 5.1 使用Paddle Inference部署
#### 5.1.1 背景

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT进行预测加速，从而实现更优的推理性能。

更多关于Paddle Inference推理引擎的介绍，可以参考Paddle Inference[官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)。
#### 5.1.2 模型和环境
---
##### 5.1.2.1 准备模型
训练的模型包括反向传播，在推理时候我们需要将模型进行精简，只保留前向传播的流程即可，将动态模型转化为用于推理的静态图模型。这里我们可以利用`export_model.py`脚本来获取该模型，其他模型可以基于该代码进行修改。
```python
cd tools
python export_model.py --save-inference-dir=model
```
**[验收]**
保存路径下面会生成3个文件，如下所示，其中在Inference推理中用到的为inference.pdiparams与inference.pdmodel。
* inference.pdiparams     : 模型参数文件
* inference.pdmodel       : 模型结构文件
* inference.pdiparams.info: 模型参数信息文件
##### 5.1.2.2 准备环境
1. 如果使用GPU的话，需要自己安转Cuda, Cudnn, TensorRT(不使用tensorRT的话可以不安装,不过建议使用加速)，同时要将每个库的动态库配置好，比如Cuda动态库设置：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
2. 去[官网](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)下载对应版本的推理框架库,假如我们本地安装的环境是:`Cuda=11.1,Cudnn=8.1,TensorRT=7.2.3.4，Python=3.6`,那么就可以下载whl包到本地，然后pip install **.whl就完成了环境的配置

#### 5.1.3 运行代码
##### 5.1.3.1整体运行代码
```python
import numpy as np
import argparse

from paddle.inference import PrecisionType
from paddle.inference import Config
from paddle.inference import create_predictor



def init_predictor(args):
    if args.model_dir is not "":
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    config.enable_memory_optim()
    if args.use_gpu:
        config.enable_use_gpu(100, 0)
        config.enable_tensorrt_engine(workspace_size=1 << 30,
                                  max_batch_size=10,
                                  min_subgraph_size=5,
                                  #precision_mode=PrecisionType.Float32,
                                  precision_mode=PrecisionType.Half,
                                  use_static=False,
                                  use_calib_mode=False)
        config.set_trt_dynamic_shape_info(
                                  min_input_shape={"input": [1, 3, 1, 1]},
                                  max_input_shape={"input": [10, 3, 1200, 1200]},
                                  optim_input_shape={"input": [1, 3, 224, 224]})
    else:
        # If not specific mkldnn, you can set the blas thread.
        # The thread num should not be greater than the number of cores in the CPU.
        config.set_cpu_math_library_num_threads(4)
        config.enable_mkldnn()

    predictor = create_predictor(config)
    return predictor


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="./model/inference.pdmodel",
        help="Model filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="./model/inference.pdiparams",
        help=
        "Parameter filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help=
        "Model dir, If you load a non-combined model, specify the directory of the model."
    )
    parser.add_argument("--use_gpu",
                        type=int,
                        default=1,
                        help="Whether use gpu.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pred = init_predictor(args)
    img = np.ones((1,3,224,224)).astype(np.float32)
    result = run(pred, [img])
    print(result)
```
##### 5.1.3.2 运行代码分解
使用 Paddle Inference 开发 Python 预测程序仅需以下五个步骤：


(1) 引用 paddle inference 预测库

```python
import paddle.inference 
```

(2) 创建配置对象，并根据需求配置，详细可参考 [Python API 文档 - Config](https://paddleinference.paddlepaddle.org.cn/api_reference/python_api_doc/Config_index.html)

```python
# 创建 config，并设置预测模型路径
config = Config(args.model_file, args.params_file)
```
此外如果使用TensorRT加速,配置详细说明可以参考[Python API 文档 - TensorRT 设置](https://paddleinference.paddlepaddle.org.cn/api_reference/python_api_doc/Config/GPUConfig.html#tensorrt)

(3) 根据Config创建预测对象predictor，详细可参考 [Python API 文档 - Predictor](https://paddleinference.paddlepaddle.org.cn/api_reference/python_api_doc/Predictor.html)

```python
predictor = create_predictor(config)
```

(4) 设置模型输入 Tensor，详细可参考 [Python API 文档 - Tensor](https://paddleinference.paddlepaddle.org.cn/api_reference/python_api_doc/Tensor.html)

```python
# 获取输入的名称
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])

# 设置输入
fake_input = np.random.randn(args.batch_size, 3, 318, 318).astype("float32")
input_handle.reshape([args.batch_size, 3, 318, 318])
input_handle.copy_from_cpu(fake_input)
```

(5) 执行预测

```python
predictor.run()
```

(5) 获得预测结果，详细可参考 [Python API 文档 - Tensor](https://paddleinference.paddlepaddle.org.cn/api_reference/python_api_doc/Tensor.html)

```python
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu() # numpy.ndarray类型
```
#### 5.1.4 常见问题
常见问题参考如下[链接](https://paddleinference.paddlepaddle.org.cn/introduction/faq.html)

<a name="6"></a>

## 6. TIPC自动化测试脚本

coming soon!

<a name="7"></a>

## 7. 参考链接与文献

1. Howard A, Sandler M, Chu G, et al. Searching for mobilenetv3[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 1314-1324.
2. vision: https://github.com/pytorch/vision
