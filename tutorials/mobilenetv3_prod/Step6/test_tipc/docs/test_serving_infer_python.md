# Linux GPU/CPU PYTHON 服务化部署测试

Linux GPU/CPU  PYTHON 服务化部署测试的主程序为`test_serving_infer_python.sh`，可以测试基于Python的模型服务化部署功能。


## 1. 测试结论汇总

- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  MobileNetV3   |  mobilenet_v3_small |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备数据

用于基础训练推理测试的数据位于`test_images/lite_data.tar`，直接解压即可（如果已经解压完成，则无需运行下面的命令）。

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
- 安装 PaddleServing 相关组件，包括serving-server、serving_client、serving-app

    ```
    # 安装0.7.0版本serving_server，用于启动服务
    wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_server_gpu-0.7.0.post102-py3-none-any.whl
    pip install paddle_serving_server_gpu-0.7.0.post102-py3-none-any.whl
    # 如果是cuda10.1环境，可以使用下面的命令安装paddle-serving-server
    # wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_server_gpu-0.7.0.post101-py3-none-any.whl
    # pip install paddle_serving_server_gpu-0.7.0.post101-py3-none-any.whl

    # 安装serving_client，用于向服务发送请求
    wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_client-0.7.0-cp37-none-any.whl
    pip install paddle_serving_client-0.7.0-cp37-none-any.whl

    # 安装serving-app
    wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_app-0.7.0-py3-none-any.whl
    pip install paddle_serving_app-0.7.0-py3-none-any.whl
    ```
**Note:** 如果要安装最新版本的PaddleServing参考[链接](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Latest_Packages_CN.md)

- 安装依赖
    ```
    pip3 install  -r requirements.txt
    ```

 ### 2.3 功能测试

 测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_serving_infer_python.sh ${your_params_file} lite_train_lite_infer
```

以`mobilenet_v3_small`的`Linux GPU/CPU PYTHON 服务化部署测试`为例，命令如下所示。

 ```bash
bash test_tipc/prepare.sh test_tipc/configs/mobilenet_v3_small/serving_infer_python.txt serving_infer
```

```bash
bash test_tipc/test_serving_infer_python.sh test_tipc/configs/mobilenet_v3_small/serving_infer_python.txt serving_infer
```

输出结果如下，表示命令运行成功。

```
Run successfully with command - python3.7 pipeline_http_client.py > ../../log/mobilenet_v3_small/serving_infer/server_infer_gpu_batchsize_1.log 2>&1 !
```

预测结果会自动保存在 `./log/mobilenet_v3_small/serving_infer/server_infer_gpu_batchsize_1.log` ，可以看到 PaddleServing 的运行结果：

```
{'err_no': 0, 'err_msg': '', 'key': ['class_id', 'prob'], 'value': ['[8]', '[0.9091243743896484]'], 'tensors': []}
```


如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
