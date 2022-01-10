# Paddle2ONNX 推理

# 目录

- [1. 简介](#1)
- [2. Paddle2ONNX推理过程](#2)
    - [2.1 准备推理环境](#2.1)
    - [2.2 模型转换](#2.2)
    - [2.3 ONNX 推理](#2.3)
- [3. FAQ](#3)

## 1. 简介
Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式，算子目前稳定支持导出 ONNX Opset 9~11，部分Paddle算子支持更低的ONNX Opset转换。

本文档主要介绍 MobileNetV3 模型如何转化为 ONNX 模型，并基于 ONNXRuntime 引擎预测。

更多细节可参考 [Paddle2ONNX官方教程](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md)

## 2. Paddle2ONNX推理过程
### 2.1 准备推理环境

需要准备 Paddle2ONNX 模型转化环境，和 ONNX 模型预测环境

- 安装 Paddle2ONNX
```
python3 -m pip install paddle2onnx
```

- 安装 ONNXRuntime
```
# 建议安装 1.9.0 版本，可根据环境更换版本号
python3 -m pip install onnxruntime==1.9.0
```

- 下载代码
```bash
git clone https://github.com/PaddlePaddle/models.git
cd models/tutorials/mobilenetv3_prod/Step6
```

### 2.2 模型转换


- Paddle 模型动转静导出

使用下面的命令完成`mobilenet_v3_net`模型的动转静导出。

```bash
#下载预训练好的参数
wget https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams
#生成推理模型
python tools/export_model.py --pretrained=./mobilenet_v3_small_pretrained.pdparams --save-inference-dir="./mobilenet_v3_small_infer" --model=mobilenet_v3_small
```

最终在`mobilenet_v3_small_infer/`文件夹下会生成下面的3个文件。

```
mobilenet_v3_small_infer
     |----inference.pdiparams     : 模型参数文件
     |----inference.pdmodel       : 模型结构文件
     |----inference.pdiparams.info: 模型参数信息文件
```

- ONNX 模型转换

使用 Paddle2ONNX 将Paddle静态图模型转换为ONNX模型格式：

```
paddle2onnx --model_dir=./mobilenetv3_model/ \
--model_filename=inference.pdmodel \
--params_filename=inference.pdiparams \
--save_file=./inference/mobilenetv3_model/model.onnx \
--opset_version=10 \
--enable_onnx_checker=True
```

执行完毕后，ONNX 模型会被保存在 `./inference/mobilenetv3_model/` 路径下


### 2.3 ONNX 推理

ONNX模型测试步骤如下：

- Step1：初始化`ONNXRuntime`库并配置相应参数, 并进行预测
- Step2：`ONNXRuntime`预测结果和`Paddle Inference`预测结果对比

对于下面的图像进行预测

<div align="center">
    <img src="../../images/demo.jpg" width=300">
</div>

执行如下命令：

```
python3 deploy/onnx_python/infer.py \
  --onnx_file ./inference/mobilenetv3_model/model.onnx \
  --params_file ./mobilenet_v3_small_pretrained.pdparams \
  --img_path ./images/demo.jpg
```

在`ONNXRuntime`输出结果如下。

```
class_id: 8, prob: 0.9091270565986633
```

表示预测的类别ID是`8`，置信度为`0.909`，该结果与基于训练引擎的结果完全一致

`ONNXRuntime`预测结果和`Paddle Inference`预测结果对比，如下。

```
The difference of results between ONNXRuntime and Paddle looks good!
max_abs_diff:  1.5646219e-07
```

从`ONNXRuntime`和`Paddle Inference`的预测结果diff可见，两者的结果几乎完全一致


## 3. FAQ
