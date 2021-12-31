# Inference 推理


# 目录

- [1. 简介]()
- [2. 推理过程]()
    - [2.1 准备推理环境]()
    - [2.2 模型动转静导出]()
    - [2.3 模型推理]()
- [3. FAQ]()


## 1. 简介

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT进行预测加速，从而实现更优的推理性能。

本文档主要基于Paddle Inference的AlexNet模型推理。

更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)。


## 2. 推理过程

### 2.1 准备推理环境

安装好PaddlePaddle即可体验Paddle Inference部署能力。


### 2.2 模型动转静导出

使用下面的命令完成`mobilenet_v3_net`模型的动转静导出。

```bash
#下载预训练好的参数
wget https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams
#生成推理模型
python tools/export_model.py --pretrained=./mobilenet_v3_small_pretrained.pdparams --save-inference-dir="./mobilenet_v3_small_infer" --model=mobilenet_v3_small
```

最终在`mobilenet_v3_small_infer/`文件夹下会生成下面的3个文件。

```
alexnet_infer
     |----inference.pdiparams     : 模型参数文件
     |----inference.pdmodel       : 模型结构文件
     |----inference.pdiparams.info: 模型参数信息文件
```

### 2.3 模型推理


```bash
python deploy/inference_python/infer.py --model-dir=./mobilenet_v3_small_infer/ --img-path=./images/demo.jpg
```

对于下面的图像进行预测

<div align="center">
    <img src="../../images/demo.jpg" width=300">
</div>

在终端中输出结果如下。

```
image_name: ./images/demo.jpg, class_id: 8, prob: 0.9091264605522156
```

表示预测的类别ID是`8`，置信度为`0.909`，该结果与基于训练引擎的结果完全一致。


## 3. FAQ
