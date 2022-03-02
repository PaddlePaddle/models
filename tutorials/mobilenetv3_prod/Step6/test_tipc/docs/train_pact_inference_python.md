# Linux GPU/CPU PACT量化训练推理测试

Linux GPU/CPU PACT量化训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 |
|  :----: |   :----:  |    :----:  |  :----:   |
|  MobileNetV3  | mobilenet_v3_small | PACT量化训练 | PACT量化训练 |


- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  MobileNetV3   |  mobilenet_v3_small |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备数据

用于量化训练推理测试的数据位于`test_images/lite_data.tar`，直接解压即可（如果已经解压完成，则无需运行下面的命令）。

```bash
tar -xf test_images/lite_data.tar
```

### 2.2 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    pip3 install paddlepaddle-gpu==2.2.0
    # 安装CPU版本的Paddle
    pip3 install paddlepaddle==2.2.0
    ```

- 安装依赖
    ```
    pip3 install  -r requirements.txt
    ```
- 安装AutoLog（规范化日志输出工具）
    ```
    pip3 install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ```


### 2.3 功能测试

以`mobilenet_v3_small`的`Linux GPU/CPU PACT量化训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/prepare.sh test_tipc/configs/mobilenet_v3_small/train_pact_infer_python.txt lite_train_lite_infer
```

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/mobilenet_v3_small/train_pact_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```
 Run successfully with command - python3.7 train.py --pact_quant --output-dir=./log/mobilenet_v3_small/lite_train_lite_infer/pact_train_gpus_0 --epochs=5   --batch-size=4!
 ......
 Run successfully with command - python3.7 train.py --test-only --pretrained=./log/mobilenet_v3_small/lite_train_lite_infer/pact_train_gpus_0/latest.pdparams!
 ......
 Run successfully with command - python3.7 tools/export_model.py --model=mobilenet_v3_small --pretrained=./log/mobilenet_v3_small/lite_train_lite_infer/pact_train_gpus_0/latest.pdparams --save-inference-dir=./log/mobilenet_v3_small/lite_train_lite_infer/pact_train_gpus_0!
 ......
 Run successfully with command - python3.7 deploy/inference_python/infer.py --use-gpu=True --model-dir=./log/mobilenet_v3_small/lite_train_lite_infer/pact_train_gpus_0 --batch-size=1   --benchmark=True > ./log/mobilenet_v3_small/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1 !
 ......
 Run successfully with command - python3.7 deploy/inference_python/infer.py --use-gpu=True --model-dir=./log/mobilenet_v3_small/lite_train_lite_infer/pact_train_gpus_0,1 --batch-size=1   --benchmark=True > ./log/mobilenet_v3_small/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1
 ......
 Run successfully with command - python3.7 deploy/inference_python/infer.py --use-gpu=False --model-dir=./log/mobilenet_v3_small/lite_train_lite_infer/pact_train_gpus_0,1 --batch-size=1   --benchmark=True > ./log/mobilenet_v3_small/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1
```

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。