# Linux端基础训练预测功能测试

Linux端基础训练预测功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。


其中裁剪、量化、蒸馏非必须。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩（单机多卡） |
|  :----  |   :----  |    :----  |  :----   |  :----   |  :----   |
|  -  | - | - | - | - | - |


- 预测相关：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，这两类模型对应的预测功能汇总如下，

| 模型类型 | device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |  :----:  |   :----:   |  :----:  |
| - |  | -/- | - | - | - |
| - | - | -/- | - | - | - |

## 2. 测试流程

运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。

### 2.1 安装依赖
- 安装PaddlePaddle >= 2.1
- 安装本项目依赖
    ```
    pip3 install  -r ../requirements.txt
    ```
- 安装autolog（规范化日志输出工具）
    ```
    git clone https://github.com/LDOUBLEV/AutoLog
    cd AutoLog
    pip3 install -r requirements.txt
    python3 setup.py bdist_wheel
    pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
    cd ../
    ```
- 安装PaddleSlim (可选)
   ```
   # 如果要测试量化、裁剪等功能，需要安装PaddleSlim
   pip3 install paddleslim
   ```


### 2.2 功能测试

先运行`prepare.sh`准备数据和模型，然后运行`test_train_inference_python.sh`进行测试，最终在`test_tipc/output`目录下生成`python_infer_*.log`格式的日志文件。

`test_train_inference_python.sh`包含5种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，这里只需要实现模式：`lite_train_lite_infer`，具体说明如下。

- 模式：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；

```shell
# 准备环境
bash test_tipc/prepare.sh ./test_tipc/configs/$model_name/train_infer_python.txt 'lite_train_lite_infer'
# 基于准备好的配置文件进行验证
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/config.txt 'lite_train_lite_infer'
```

运行相应指令后，在`test_tipc/output`文件夹下自动会保存运行日志。如`lite_train_lite_infer`模式下，会运行`训练+inference`的链条，因此，在`test_tipc/output`文件夹有以下文件：

```
test_tipc/output/
|- results_python.log    # 运行指令状态的日志
|- norm_train_gpus_0_autocast_null/  # GPU 0号卡上正常训练的训练日志和模型保存文件夹
......
```

其他模式中其中`results_python.log`中包含了每条指令的运行状态，如果运行成功会输出：

```
Run successfully with xxxxx
......
```

如果运行失败，会输出：

```
Run failed with xxxxx
......
```

可以很方便的根据`results_python.log`中的内容判定哪一个指令运行错误。


## 3. 更多教程
本文档为功能测试用，更丰富的训练预测使用教程请参考：

- [模型训练](../../README.md)  
- [基于Python预测引擎推理](../../deploy/pdinference/README.md)
