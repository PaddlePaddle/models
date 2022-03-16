# Lite cpp arm cpu 推理功能测试

Lite cpp arm cpu 推理测试的主程序为`test_lite_arm_cpu_cpp.sh`，可以测试基于基于 ARM CPU 的 cpp 模型推理功能。

## 1. 测试结论汇总

| 算法名称 | 模型名称 | num_threads | precision | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  MobileNetV3   |  mobilenet_v3_small |  1 | FP32 | 1 |


## 2. 测试流程

### 2.1 准备环境

Lite cpp arm cpu 推理的环境准备可以参考[]()。


### 2.2 准备模型、数据、预测库

`prepare_lite_arm_cpu_cpp.sh` 中集成了准备模型、数据、预测库的相关命令。

```bash
bash test_tipc/prepare_lite_arm_cpu_cpp.sh ${your_config_file}
```

以`mobilenet_v3_small`的`Lite cpp arm cpu 推理功能测试`为例，命令如下所示。

```bash
bash test_tipc/prepare_lite_arm_cpu_cpp.sh test_tipc/configs/mobilenet_v3_small/lite_arm_cpu_cpp.txt
```

### 2.3 功能测试

`test_lite_arm_cpu_cpp.sh` 中包含了测试的内容，执行以下命令即可完成测试。

`bash test_tipc/test_lite_arm_cpu_cpp.sh ${your_config_file}`

以`mobilenet_v3_small`的`Lite cpp arm cpu 推理功能测试`为例，命令如下所示。

```bash
`bash test_tipc/test_lite_arm_cpu_cpp.sh test_tipc/configs/mobilenet_v3_small/lite_arm_cpu_cpp.txt`
```

输出结果如下，表示命令运行成功。

```bash
 Run successfully with command - adb shell 'export LD_LIBRARY_PATH=/data/local/tmp/arm_cpu/; /data/local/tmp/arm_cpu/mobilenet_v3 /data/local/tmp/arm_cpu/config.txt /data/local/tmp/arm_cpu/demo.jpg'  > ./output/lite_mobilenet_v3_small.nb_runtime_device_arm_cpu_precision_FP32_batchsize_1_threads_1.log 2>&1!
```


可以打印参数设置信息（运行设备、线程数等），模型信息（模型名称、精度等），数据信息（batchsize等），性能信息（预处理耗时、推理耗时、后处理耗时），如下图所示

<div align="center">
    <img src="../../../../tipc/train_infer_python/images/lite_cpp_arm_cpu_autolog_demo.png">
</div>

该信息可以在运行log中查看，以`mobilenet_v3_small`为例，log位置在`./output/lite_mobilenet_v3_small.nb_runtime_device_arm_cpu_precision_FP32_batchsize_1_threads_1.log`。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
