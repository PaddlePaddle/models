# MobileNetV3

## 目录


- [1. 简介](#1)
- [2. 离线量化](#2)
    - [2.1 准备Inference模型及环境](#2.1)
    - [2.2 开始离线量化](#2.2)
    - [2.3 验证推理结果](#2.3)
- [3. FAQ](#3)


<a name="1"></a>

## 1. 简介

Paddle中静态离线量化，使用少量校准数据计算量化因子，可以快速将FP32模型量化成低比特模型（比如最常用的int8量化）。使用该量化模型进行预测，可以减少计算量、降低计算内存、减小模型大小。

本文档主要基于Paddle的MobileNetV3模型进行离线量化。

更多关于Paddle 模型离线量化的介绍，可以参考[Paddle 离线量化官网教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/static/quant/quantization_api.rst#quant_post_static)。


<a name="2"></a>

## 2. 离线量化

<a name="2.1"></a>

### 2.1 准备Inference模型及环境

由于离线量化直接使用Inference模型进行量化，不依赖模型组网，所以需要提前准备好Inference模型.
我们准备好了动转静后的MobileNetv3 small的Inference模型，可以从[mobilenet_v3_small_infer](https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_infer.tar)直接下载。

```shell
wget https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_infer.tar
tar -xf mobilenet_v3_small_infer.tar
```

也可以按照[MobileNetv3 动转静流程](xxx)，将MobileNetv3 small的模型转成Inference模型。

<a name="2.2"></a>

环境准备：

- 安装PaddleSlim：
```shell
pip install paddleslim==2.2.1
```

- 安装PaddlePaddle：
```shell
pip install paddlepaddle-gpu==2.2.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

- 准备数据：

请参考[数据准备文档](https://github.com/PaddlePaddle/models/tree/release/2.2/tutorials/mobilenetv3_prod/Step6#32-%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE)。

### 2.2 开始离线量化

启动离线量化：

```bash
python post_quant.py --model_path=mobilenet_v3_small_infer/ \
            --model_filename=inference.pdmodel \
            --params_filename=inference.pdiparams  \
            --data_dir=/path/dataset/ILSVRC2012/ \
            --use_gpu=True \
            --batch_size=32 \
            --batch_num=20
```

部分离线量化日志如下：

```
Thu Dec 30 12:36:17-INFO: Collect quantized variable names ...
Thu Dec 30 12:36:17-INFO: Preparation stage ...
Thu Dec 30 12:36:27-INFO: Run batch: 0
Thu Dec 30 12:37:10-INFO: Run batch: 5
Thu Dec 30 12:37:43-INFO: Finish preparation stage, all batch:10
Thu Dec 30 12:37:43-INFO: Sampling stage ...
Thu Dec 30 12:38:10-INFO: Run batch: 0
Thu Dec 30 12:39:03-INFO: Run batch: 5
Thu Dec 30 12:39:46-INFO: Finish sampling stage, all batch: 10
Thu Dec 30 12:39:46-INFO: Calculate hist threshold ...
Thu Dec 30 12:39:47-INFO: Update the program ...
Thu Dec 30 12:39:49-INFO: The quantized model is saved in output/mv3_int8_infer
```

离线量化完成后，会在`output_dir`中生成量化后的Inference模型。

<a name="2.3"></a>

### 2.3 验证推理结果

- 量化推理模型重新命名：

需要将`__model__`重命名为`inference.pdmodel`，将`__params__`重命名为`inference.pdiparams`。

正确的命名如下：
```shell
output/mv3_int8_infer/
    |----inference.pdiparams     : 模型参数文件(原__params__文件)
    |----inference.pdmodel       : 模型结构文件(原__model__文件)
```

- 使用Paddle Inference测试模型推理结果是否正确：

具体测试流程请参考[Inference推理文档](https://github.com/PaddlePaddle/models/blob/release/2.2/tutorials/mobilenetv3_prod/Step6/deploy/inference_python/README.md)

如果您希望验证量化模型的在全量验证集上的精度，也可以按照下面的步骤进行操作:

使用如下命令验证MobileNetv3 small模型的精度：

- FP32模型：
```bash
python eval.py --model_path=mobilenet_v3_small_infer/ \
        --model_filename=inference.pdmodel \
        --params_filename=inference.pdiparams \
        --data_dir=/path/dataset/ILSVRC2012/ \
        --batch_size=128 \
        --use_gpu=True
```

FP32模型精度验证日志如下：

```
batch_id 300, acc1 0.602, acc5 0.825, avg time 0.00005 sec/img
batch_id 310, acc1 0.602, acc5 0.825, avg time 0.00005 sec/img
batch_id 320, acc1 0.602, acc5 0.825, avg time 0.00005 sec/img
batch_id 330, acc1 0.602, acc5 0.825, avg time 0.00005 sec/img
batch_id 340, acc1 0.601, acc5 0.825, avg time 0.00005 sec/img
batch_id 350, acc1 0.601, acc5 0.825, avg time 0.00005 sec/img
batch_id 360, acc1 0.602, acc5 0.826, avg time 0.00005 sec/img
batch_id 370, acc1 0.602, acc5 0.826, avg time 0.00005 sec/img
batch_id 380, acc1 0.602, acc5 0.825, avg time 0.00005 sec/img
batch_id 390, acc1 0.601, acc5 0.825, avg time 0.00005 sec/img
End test: test image 50000.0
test_acc1 0.6015, test_acc5 0.8253, avg time 0.00005 sec/img
```

- 量化模型：
```shell
python eval.py --model_path=output/mv3_int8_infer/ \
        --model_filename=__model__ \
        --params_filename=__params__ \
        --data_dir=/path/dataset/ILSVRC2012/ \
        --batch_size=128 \
        --use_gpu=True
```

量化后模型精度验证日志如下：

```
batch_id 300, acc1 0.564, acc5 0.800, avg time 0.00006 sec/img
batch_id 310, acc1 0.562, acc5 0.798, avg time 0.00006 sec/img
batch_id 320, acc1 0.560, acc5 0.796, avg time 0.00006 sec/img
batch_id 330, acc1 0.556, acc5 0.792, avg time 0.00006 sec/img
batch_id 340, acc1 0.554, acc5 0.792, avg time 0.00006 sec/img
batch_id 350, acc1 0.552, acc5 0.790, avg time 0.00006 sec/img
batch_id 360, acc1 0.550, acc5 0.789, avg time 0.00006 sec/img
batch_id 370, acc1 0.551, acc5 0.789, avg time 0.00006 sec/img
batch_id 380, acc1 0.551, acc5 0.789, avg time 0.00006 sec/img
batch_id 390, acc1 0.553, acc5 0.790, avg time 0.00006 sec/img
End test: test image 50000.0
test_acc1 0.5530, test_acc5 0.7905, avg time 0.00006 sec/img
```

<a name="3"></a>

## 3. FAQ
