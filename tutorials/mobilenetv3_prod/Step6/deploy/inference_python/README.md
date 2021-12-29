### 1 使用Paddle Inference部署
#### 1.1 背景

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT进行预测加速，从而实现更优的推理性能。

更多关于Paddle Inference推理引擎的介绍，可以参考Paddle Inference[官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)。
#### 1.2 模型和环境
---
##### 1.2.1 准备模型
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
##### 1.2.2 准备环境
1. 如果使用GPU的话，需要自己安转Cuda, Cudnn, TensorRT(不使用tensorRT的话可以不安装,不过建议使用加速)，同时要将每个库的动态库配置好，比如Cuda动态库设置：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
2. 去[官网](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)下载对应版本的推理框架库,假如我们本地安装的环境是:`Cuda=11.1,Cudnn=8.1,TensorRT=7.2.3.4，Python=3.6`,那么就可以下载whl包到本地，然后pip install **.whl就完成了环境的配置

#### 1.3 运行代码
##### 1.3.1整体运行代码
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
##### 1.3.2 运行代码分解
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
#### 1.4 常见问题
常见问题参考如下[链接](https://paddleinference.paddlepaddle.org.cn/introduction/faq.html)
