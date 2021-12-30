### 使用Paddle Inference部署
#### 背景

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT进行预测加速，从而实现更优的推理性能。

更多关于Paddle Inference推理引擎的介绍，可以参考Paddle Inference[官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)。
#### 模型和环境
---
##### 准备模型
训练的模型包括反向传播，在推理时候我们需要将模型进行精简，只保留前向传播的流程即可，将动态模型转化为用于推理的静态图模型。这里我们可以利用`export_model.py`脚本来获取该模型，其他模型可以基于该代码进行修改，本教程模型获取的脚本为[export_model.py](https://github.com/PaddlePaddle/models/blob/79e14a5935372af1848921c4e12122f0b94c5a50/tutorials/mobilenetv3_prod/Step6/tools/export_model.py)
```python
python export_model.py --save-inference-dir=model
```
**[验收]**
保存路径下面会生成3个文件，如下所示，其中在Inference推理中用到的为inference.pdiparams与inference.pdmodel。
* inference.pdiparams     : 模型参数文件
* inference.pdmodel       : 模型结构文件
* inference.pdiparams.info: 模型参数信息文件
##### 准备环境
1. 如果使用GPU的话，需要自己安转Cuda, Cudnn, TensorRT(不使用tensorRT的话可以不安装,不过建议使用加速)，同时要将每个库的动态库配置好，比如Cuda动态库设置：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
2. 去[官网](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)下载对应版本的推理框架库,假如我们本地安装的环境是:`Cuda=11.1,Cudnn=8.1,TensorRT=7.2.3.4，Python=3.6`,那么就可以下载whl包到本地，然后pip install **.whl就完成了环境的配置

#### 运行代码
##### 整体运行代码
```python
python ./infer.py --model-dir=./model --img-path=../images/demo.jpg
```
对于下面的图像进行预测

<div align="center">
    <img src="../../images/demo.jpg" width=300">
</div>

在终端中输出结果如下。

```
image_name: ../../images/demo.jpg, class_id: 812, prob: 0.001000000862404704
```

表示预测的类别ID是`812`，置信度为`0.001`，该模型未经过训练，所以1000类每个概率都是0.001。

##### 运行代码分解
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
#### 常见问题
常见问题参考如下[链接](https://paddleinference.paddlepaddle.org.cn/introduction/faq.html)
