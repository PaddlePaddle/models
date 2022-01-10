# Paddle2ONNX功能开发文档

# 目录

- [1. 简介](#1)
- [2. Paddle2ONNX推理过程开发](#2)
    - [2.1 准备环境](#2.1)
    - [2.2 转换模型](#2.2)
    - [2.3 开发数据预处理程序](#2.3)
    - [2.4 开发ONNX模型推理程序](#2.4)
    - [2.5 开发数据后处理程序](#2.5)
    - [2.6 验证ONNX推理结果正确性](#2.6)
- [3. FAQ](#3)

## 1. 简介
Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式，算子目前稳定支持导出 ONNX Opset 9~11，部分Paddle算子支持更低的ONNX Opset转换。

本文档主要介绍飞桨模型如何转化为 ONNX 模型，并基于 ONNXRuntime 引擎的推理过程开发。

更多细节可参考 [Paddle2ONNX官方教程](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md)

## 2. Paddle2ONNX推理过程开发

### 2.1 准备环境

**【数据】**

从验证集或者测试集中抽出至少一张图像，用于后续推理过程验证。

**【环境】**

需要准备 Paddle2ONNX 模型转化环境，和 ONNX 模型预测环境

- 安装 Paddle2ONNX
```
python3 -m pip install paddle2onnx
```

- 安装 onnxruntime
```
# 建议安装 1.9.0 版本，可根据环境更换版本号
python3 -m pip install onnxruntime==1.9.0
```

### 2.2 转换模型

- Paddle 模型动转静导出

**【基本内容】**

`模型动转静`方法可以将训练得到的动态图模型转化为用于推理的静态图模型，具体可参考[Linux GPU/CPU 模型推理开发文档](https://github.com/PaddlePaddle/models/blob/release%2F2.2/docs/tipc/train_infer_python/infer_python.md#2.2)中第2.2章节。

**【实战】**

参考MobileNetV3的paddle2onnx [说明文档](../../mobilenetv3_prod/Step6/deploy/onnx_python/README.md)中的第2.2章节


- ONNX 模型转换

**【基本内容】**

使用 Paddle2ONNX 将Paddle静态图模型转换为ONNX模型格式：

```
paddle2onnx --model_dir=${your_inference_model_dir}
--model_filename=${your_pdmodel_file}
--params_filename=${your_pdiparams_file}
--save_file=${output_file}
--opset_version=10
--enable_onnx_checker=True
```

- 参数说明：
  - ${your_inference_model_dir}指的是Paddle模型所在目录.
  - ${your_pdmodel_file}指的是网络结构的文件.
  - ${your_pdiparams_file}指的是模型参数的文件.
  - ${output_file}指的是需要导出的onnx模型.
  - ${opset_version}指的是ONNX Opset，目前稳定支持9～11，默认是10.
  - ${enable_onnx_checker}指的是否检查导出为ONNX模型的正确性.

更多关于参数的用法，可参考 [Paddle2ONNX官方教程](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md)

**【实战】**

参考MobileNetV3的paddle2onnx [说明文档](../../mobilenetv3_prod/Step6/deploy/onnx_python/README.md)中的第2.2章节

**【核验】**

执行完毕后，将产出${output_file} ONNX 模型文件，其中文件后缀为.onnx。


### 2.3 开发数据预处理程序

**【基本内容】**

读取指定图像，对其进行数据变换，转化为符合模型推理所需要的输入格式。

使用ONNX模型进行推理时，使用的数据预处理方法，和使用paddle inference进行推理时的预处理方法一样。


### 2.4 开发ONNX模型推理程序

ONNX作为开源的神经网络交换格式，得到大多数推理引擎的部署支持。在本文档中我们采用微软开源的onnxruntime推理引擎，进行转换后模型的正确性较验。

**【基本内容】**

使用ONNXRuntime测试ONNX模型，确保模型精度符合预期，初始化`ONNXRuntime`库并配置相应参数, 并进行预测

```
from onnxruntime import InferenceSession

# 加载ONNX模型
sess = InferenceSession('${your_onnx_model_name}.onnx')

# 模型预测
ort_outs = sess.run(output_names=None, input_feed={sess.get_inputs()[0].name: ${input_data})
```

**【注意事项】**

${input_data} 是预处理后的数据，作为网络的输入，数据是ndarray类型。


### 2.5 开发数据后处理程序

在完成ONNX模型进行推理后，基于不同的任务，需要对网络的输出进行后处理，这部分和使用paddle inference进行模型推理后的后处理方法一样。


### 2.6 验证ONNX推理结果正确性

**【基本内容】**

`ONNXRuntime`预测结果和`paddle inference`预测结果对比

```
import os
import time
import paddle

# 从模型代码中导入模型
from paddlevision.models import mobilenet_v3_small

# 实例化模型
model = mobilenet_v3_small('${your_paddle_model_name}.pdparams')

# 将模型设置为推理状态
model.eval()

# 对比ONNXRuntime和Paddle预测的结果
paddle_outs = model(paddle.to_tensor(${input_data}))

diff = ort_outs[0] - paddle_outs.numpy()
max_abs_diff = np.fabs(diff).max()
if max_abs_diff < 1e-05:
    print("The difference of results between ONNXRuntime and Paddle looks good!")
else:
    relative_diff = max_abs_diff / np.fabs(paddle_outs.numpy()).max()
    if relative_diff < 1e-05:
        print("The difference of results between ONNXRuntime and Paddle looks good!")
    else:
        print("The difference of results between ONNXRuntime and Paddle looks bad!")
    print('relative_diff: ', relative_diff)
print('max_abs_diff: ', max_abs_diff)

```

**【注意事项】**

${input_data} 是预处理后的数据，和 ONNXRuntime 的输入一样。

ort_outs 是 ONNXRuntime 的输出结果

paddlevision 模块位于MobileNetV3_prod/Step6目录下

**【实战】**

参考MobileNetV3的paddle2onnx [说明文档](../../mobilenetv3_prod/Step6/deploy/onnx_python/README.md)中的第2.3章节

## 3. FAQ

如果您在使用该文档完成paddle模型转ONNX的过程中遇到问题，可以给在[这里](https://github.com/PaddlePaddle/Paddle2ONNX/issues)提一个ISSUE，我们会高优跟进。
