# MobileNetV3

## 目录


- [1. 准备数据与环境](#1)
    - [1.1 准备环境](#1.1)
    - [1.2 准备数据](#1.2)
    - [1.3 准备模型](#1.3)
- [2. 开始使用](#2)
    - [2.1 模型训练](#2.1)
    - [2.2 模型评估](#2.2)
    - [2.3 模型预测](#2.3)
- [3. 模型推理部署](#3)

<a name="1"></a>

本文档主要介绍MobileNetV3模型在Windows平台的推理开发流程，有关MobileNetV3模型和数据集的介绍参考 [首页](../REDAME.md)。需要注意，在Windows平台上执行命令和Linux平台略有不同，主要体现在：下载与解压数据、设置环境变量、数据加载等方面。此外Windows平台只支持单卡的训练与预测。
## 1. 准备环境与数据


<a name="1.1"></a>

### 1.1 准备环境

* 下载代码

```bash
git clone https://github.com/PaddlePaddle/models.git
cd models/tutorials/mobilenetv3_prod/Step6
```

* 安装paddlepaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。

```bash
# 需要安装2.2及以上版本的Paddle
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
W0119 16:25:14.953202  7104 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 6.1, Driver API Version: 11.4, Runtime API Version: 10.2
W0119 16:25:14.953202  7104 device_context.cc:465] device: 0, cuDNN Version: 7.6.
PaddlePaddle works well on 1 GPU.
PaddlePaddle works well on 1 GPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 安装requirements

```bash
pip install -r requirements.txt
```

<a name="1.2"></a>

### 1.2 准备数据

如果您已经下载好ImageNet1k数据集，那么该步骤可以跳过，如果您没有，则可以从[ImageNet官网](https://image-net.org/download.php)申请下载。

如果只是希望快速体验模型训练功能，则可以直接解压`test_images/lite_data.tar`，其中包含16张训练图像以及16张验证图像。

```bash
python -c "import shutil;shutil.unpack_archive('test_images/lite_data.tar', extract_dir='./',format='tar')"
```

执行该命令后，会在当前路径下解压出对应的数据集文件夹lite_data


<a name="1.3"></a>

### 1.3 准备模型

如果您希望直接体验评估或者预测推理过程，可以使用下面的命令下载 MobileNetV3 预训练模型，直接体验模型评估、预测、推理部署等内容。

```bash
# 下载预训练模型
pip install wget
python -c "import wget;wget.download('https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams')"
# 下载推理模型
# coming soon!
```


<a name="2"></a>

## 2. 开始使用

<a name="2.1"></a>

### 2.1 模型训练

* 单机单卡训练

```bash
# 在Windows平台，DataLoader只支持单进程模式，因此需要设置 workers 为0
set CUDA_VISIBLE_DEVICES=0
python train.py --data-path=./ILSVRC2012 --lr=0.1 --batch-size=256 --workers=0
```

部分训练日志如下所示。

```
[Epoch 1, iter: 4780] top1: 0.10312, top5: 0.27344, lr: 0.01000, loss: 5.34719, avg_reader_cost: 0.03644 sec, avg_batch_cost: 0.05536 sec, avg_samples: 64.0, avg_ips: 1156.08863 images/sec.
[Epoch 1, iter: 4790] top1: 0.08750, top5: 0.24531, lr: 0.01000, loss: 5.28853, avg_reader_cost: 0.05164 sec, avg_batch_cost: 0.06852 sec, avg_samples: 64.0, avg_ips: 934.08427 images/sec.
```

**注意**：目前Windows平台只支持单卡训练与预测。

更多配置参数可以参考[train.py](./train.py)的`get_args_parser`函数。

<a name="2.2"></a>

### 2.2 模型评估

该项目中，训练与评估脚本相同，指定`--test-only`参数即可完成预测过程。

```bash
# 在Windows平台，DataLoader只支持单进程模式，因此需要设置 workers 为0
python train.py --test-only --data-path=./ILSVRC2012 --pretrained=./mobilenet_v3_small_pretrained.pdparams --workers=0
```

期望输出如下。

```
Test:  [   0/1563]  eta: 1:14:20  loss: 1.0456 (1.0456)  acc1: 0.7812 (0.7812)  acc5: 0.9062 (0.9062)  time: 2.8539  data: 2.8262
...
Test:  [1500/1563]  eta: 0:00:05  loss: 1.2878 (1.9196)  acc1: 0.7344 (0.5639)  acc5: 0.8750 (0.7893)  time: 0.0623  data: 0.0534
Test: Total time: 0:02:05
 * Acc@1 0.564 Acc@5 0.790
```

<a name="2.3"></a>

### 2.3 模型预测

* 使用GPU预测

```bash
python tools/predict.py --pretrained=./mobilenet_v3_small_pretrained.pdparams --img-path=images/demo.jpg
```

对于下面的图像进行预测

<div align="center">
    <img src="../images/demo.jpg" width=300">
</div>

最终输出结果为`class_id: 8, prob: 0.9091238975524902`，表示预测的类别ID是`8`，置信度为`0.909`。

* 使用CPU预测

```bash
python tools/predict.py --pretrained=./mobilenet_v3_small_pretrained.pdparams --img-path=images/demo.jpg --device=cpu
```

<a name="3"></a>

## 3. 模型推理部署

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT进行预测加速，从而实现更优的推理性能。

更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)。

本小节教程主要基于Paddle Inference的mobilenet_v3_small模型推理。假定已安装好PaddlePaddle，当前路径为 `xx/models/tutorials/mobilenetv3_prod/Step6`。


### 3.1 模型动转静导出

使用下面的命令，将1.3小节中下载的`mobilenet_v3_net`模型参数进行动转静导出。

```bash
#生成推理模型
python tools/export_model.py --pretrained=./mobilenet_v3_small_pretrained.pdparams --save-inference-dir="./mobilenet_v3_small_infer" --model=mobilenet_v3_small
```

在`mobilenet_v3_small_infer/`文件夹下会生成下面的3个文件。

```
mobilenet_v3_small_infer
     |----inference.pdiparams     : 模型参数文件
     |----inference.pdmodel       : 模型结构文件
     |----inference.pdiparams.info: 模型参数信息文件
```

### 3.2 模型推理


```bash
python deploy/inference_python/infer.py --model-dir=./mobilenet_v3_small_infer/ --img-path=./images/demo.jpg
```

对于下面的图像进行预测

<div align="center">
    <img src="../images/demo.jpg" width=300">
</div>

在终端中输出结果如下。

```
image_name: ./images/demo.jpg, class_id: 8, prob: 0.9091264605522156
```

表示预测的类别ID是`8`，置信度为`0.909`，该结果与基于训练引擎的结果完全一致。

