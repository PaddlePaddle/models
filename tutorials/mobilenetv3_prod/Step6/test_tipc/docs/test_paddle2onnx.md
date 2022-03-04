# Paddle2ONNX 测试

Paddle2ONNX 测试的主程序为`test_paddle2onnx.sh`，可以测试基于Paddle2ONNX的模型转换和onnx预测功能。


## 1. 测试结论汇总

- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  MobileNetV3   |  mobilenet_v3_small |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备数据

用于基础训练推理测试的数据位于`test_images/lite_data.tar`，直接解压即可（如果已经解压完成，则无需运行下面的命令）。

```
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
    pip3 install  -r requirements.txt
    ```

- 安装 Paddle2ONNX
    ```
    pip install paddle2onnx
    ```

- 安装 ONNXRuntime
    ```
    # 建议安装 1.9.0 版本，可根据环境更换版本号
    pip install onnxruntime==1.9.0
    ```


### 2.3 功能测试

测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_paddle2onnx.sh ${your_params_file} paddle2onnx_infer
```

以`mobilenet_v3_small`的`Paddle2ONNX 测试`为例，命令如下所示。

 ```bash
bash test_tipc/prepare.sh test_tipc/configs/mobilenet_v3_small/paddle2onnx_infer_python.txt paddle2onnx_infer
```

```bash
bash test_tipc/test_paddle2onnx.sh test_tipc/configs/mobilenet_v3_small/paddle2onnx_infer_python.txt paddle2onnx_infer
```

输出结果如下，表示命令运行成功。

```
Run successfully with command -  paddle2onnx --model_dir=./inference/mobilenet_v3_small_infer/ --model_filename=inference.pdmodel --params_filename=inference.pdiparams --save_file=./inference/mobilenet_v3_small_onnx/model.onnx --opset_version=10 --enable_onnx_checker=True!

Run successfully with command - python3.7 deploy/onnx_python/infer.py --img_path=./lite_data/test/demo.jpg --onnx_file=./inference/mobilenet_v3_small_onnx/model.onnx > ./log/mobilenet_v3_small//paddle2onnx_infer_cpu.log 2>&1 !  
```

预测结果会自动保存在 `./log/mobilenet_v3_small/paddle2onnx_infer_cpu.log` ，可以看到onnx运行结果：
```
ONNXRuntime predict:
class_id: 8, prob: 0.9091271758079529
```

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
