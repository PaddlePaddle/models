# Paddle2ONNX功能开发文档

# 目录

- [1. 简介](#1---)
- [2. Paddle2ONNX功能开发](#2---)
    - [2.1 环境准备](#2.1---)
    - [2.2 模型转换](#2.2---)
    - [2.3 ONNX 预测](#2.3---)
- [3. FAQ](#3---)

## 1. 简介
Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式，算子目前稳定支持导出 ONNX Opset 9~11，部分Paddle算子支持更低的ONNX Opset转换。

本文档主要介绍 MobileNetV3 模型如何转化为 ONNX 模型，并基于 ONNX 引擎预测。

更多细节可参考 [Paddle2ONNX官方教程](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md)

## 2. Paddle2ONNX功能开发
### 2.1 环境准备

需要准备 Paddle2ONNX 模型转化环境，和 ONNX 模型预测环境

- 安装 Paddle2ONNX
```
python3 -m pip install paddle2onnx
```

- 安装 ONNX
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

使用下面的命令完成`MobileNetV3`模型的动转静导出。

```bash
python3 ./tools/export_model.py --pretrained=./mobilenet_v3_small_pretrained.pdparams --save-inference-dir=./mobilenetv3_model
```
最终在`mobilenetv3_model/`文件夹下会生成下面的3个文件。

```
mobilenetv3_model
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


### 2.3 ONNX 预测

ONNX模型测试：


```
import time
from PIL import Image
from onnxruntime import InferenceSession
from presets import ClassificationPresetEval

# 加载ONNX模型
sess = InferenceSession('./inference/mobilenetv3_model/model.onnx')

# define transforms
input_shape = sess.get_inputs()[0].shape[2:]
eval_transforms = ClassificationPresetEval(crop_size=input_shape,
                                           resize_size=256)
# 准备输入
with open('./images/demo.jpg', 'rb') as f:
    img = Image.open(f).convert('RGB')

img = eval_transforms(img)
img = img.expand([1] + img.shape)

# 模型预测
start = time.time()
ort_outs = sess.run(output_names=None,
                    input_feed={sess.get_inputs()[0].name: img.numpy()})
end = time.time()

output = ort_outs[0]
class_id = output.argmax()
prob = output[0][class_id]
print(f"class_id: {class_id}, prob: {prob}")
print('ONNXRuntime predict time: %.04f s' % (end - start))

```

对于下面的图像进行预测

<div align="center">
    <img src="../../mobilenetv3_prod/Step6/images/demo.jpg" width=300">
</div>

在终端中输出结果如下。

```
class_id: 8, prob: 0.9091273546218872
ONNXRuntime predict time: 0.0047 s
```

表示预测的类别ID是`8`，置信度为`0.909`，该结果与基于训练引擎的结果完全一致

## 3. FAQ
