# Linux GPU/CPU 混合精度训练推理测试

Linux GPU/CPU 混合精度训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 |
|  :----: |   :----:  |    :----:  |  :----:   |
|  MobileNetV3  | mobilenet_v3_small | 混合精度训练 | 混合精度训练 |


- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  MobileNetV3   |  mobilenet_v3_small |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备数据

用于混合精度训练推理测试的数据位于`test_images/lite_data.tar`，直接解压即可（如果已经解压完成，则无需运行下面的命令）。

```bash
tar -xf test_images/lite_data.tar
```

### 2.2 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    pip install paddlepaddle-gpu==2.2.0
    # 安装CPU版本的Paddle
    pip install paddlepaddle==2.2.0
    ```

- 安装依赖
    ```
    pip install  -r requirements.txt
    ```
- 安装AutoLog（规范化日志输出工具）
    ```
    pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ```

### 2.3 功能测试


测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_lite_infer
```

以`mobilenet_v3_small`的`Linux GPU/CPU 混合精度训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/prepare.sh test_tipc/configs/mobilenet_v3_small/train_amp_infer_python.txt lite_train_lite_infer
```

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/mobilenet_v3_small/train_amp_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```bash
Run successfully with command - python3.7 train.py --amp_level=O1 --output-dir=./log/mobilenet_v3_small/lite_train_lite_infer/amp_train_gpus_0 --epochs=5   --batch-size=4!
......
Run successfully with command - python3.7 deploy/inference_python/infer.py --use-gpu=False --model-dir=./log/mobilenet_v3_small/lite_train_lite_infer/amp_train_gpus_0,1 --batch-size=1   --benchmark=True > ./log/mobilenet_v3_small/lite_train_lite_infer/amp_train_python_infer_cpu_batchsize_1.log 2>&1 !
```

在开启benchmark选项时，可以得到测试的详细数据，包含运行环境信息（系统版本、CUDA版本、CUDNN版本、驱动版本），Paddle版本信息，参数设置信息（运行设备、线程数、是否开启内存优化等），模型信息（模型名称、精度），数据信息（batchsize、是否为动态shape等），性能信息（CPU/GPU的占用、运行耗时、预处理耗时、推理耗时、后处理耗时），内容如下所示：

```
[2022/03/03 04:21:20] root INFO: ---------------------- Env info ----------------------
[2022/03/03 04:21:20] root INFO:  OS_version: Ubuntu 16.04
[2022/03/03 04:21:20] root INFO:  CUDA_version: 10.2.89
[2022/03/03 04:21:20] root INFO:  CUDNN_version: 7.6.5
[2022/03/03 04:21:20] root INFO:  drivier_version: 440.64.00
[2022/03/03 04:21:20] root INFO: ---------------------- Paddle info ----------------------
[2022/03/03 04:21:20] root INFO:  paddle_version: 2.2.2
[2022/03/03 04:21:20] root INFO:  paddle_commit: b031c389938bfa15e15bb20494c76f86289d77b0
[2022/03/03 04:21:20] root INFO:  log_api_version: 1.0
[2022/03/03 04:21:20] root INFO: ----------------------- Conf info -----------------------
[2022/03/03 04:21:20] root INFO:  runtime_device: cpu
[2022/03/03 04:21:20] root INFO:  ir_optim: True
[2022/03/03 04:21:20] root INFO:  enable_memory_optim: True
[2022/03/03 04:21:20] root INFO:  enable_tensorrt: False
[2022/03/03 04:21:20] root INFO:  enable_mkldnn: False
[2022/03/03 04:21:20] root INFO:  cpu_math_library_num_threads: 1
[2022/03/03 04:21:20] root INFO: ----------------------- Model info ----------------------
[2022/03/03 04:21:20] root INFO:  model_name: classification
[2022/03/03 04:21:20] root INFO:  precision: fp32
[2022/03/03 04:21:20] root INFO: ----------------------- Data info -----------------------
[2022/03/03 04:21:20] root INFO:  batch_size: 1
[2022/03/03 04:21:20] root INFO:  input_shape: dynamic
[2022/03/03 04:21:20] root INFO:  data_num: 1
[2022/03/03 04:21:20] root INFO: ----------------------- Perf info -----------------------
[2022/03/03 04:21:20] root INFO:  cpu_rss(MB): 228.7539, gpu_rss(MB): None, gpu_util: None%
[2022/03/03 04:21:20] root INFO:  total time spent(s): 0.2199
[2022/03/03 04:21:20] root INFO:  preprocess_time(ms): 18.5826, inference_time(ms): 201.2458, postprocess_time(ms): 0.0784
```

该信息可以在运行log中查看，以`mobilenet_v3_small`为例，log位置在`./log/mobilenet_v3_small/lite_train_lite_infer/python_infer_gpu_batchsize_1.log`。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
