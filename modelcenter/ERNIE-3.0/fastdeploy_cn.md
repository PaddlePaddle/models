## 0. 全场景高性能AI推理部署工具 FastDeploy
FastDeploy 是一款**全场景、易用灵活、极致高效**的AI推理部署工具。提供开箱即用的**云边端**部署体验, 支持超过 150+ Text, Vision, Speech和跨模态模型，实现了AI模型**端到端的优化加速**。目前支持的硬件包括 **X86 CPU、NVIDIA GPU、ARM CPU、XPU、NPU、IPU**等10类云边端的硬件，通过一行代码切换不同推理后端和硬件。

使用 FastDeploy 3步即可搞定AI模型部署：（1）安装FastDeploy预编译包（2）调用FastDeploy的API实现部署代码 （3）推理部署。

**注** : 本文档下载 FastDeploy 示例来完成高性能部署体验；仅展示X86 CPU、NVIDIA GPU的推理，且默认已经准备好GPU环境（如 CUDA >= 11.2等），如需要部署其他硬件或者完整了解 FastDeploy 部署能力，请参考 [FastDeploy的GitHub仓库](https://github.com/PaddlePaddle/FastDeploy)


## 1. 安装FastDeploy预编译包
```
pip install fastdeploy-gpu-python==0.0.0 -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html
```
## 2. 运行部署示例
```
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/text/ernie-3.0/python

# 下载AFQMC数据集的微调后的ERNIE 3.0模型
wget https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-afqmc.tgz
tar xvfz ernie-3.0-medium-zh-afqmc.tgz

# CPU 推理
python seq_cls_infer.py --device cpu --model_dir ernie-3.0-medium-zh-afqmc

# GPU 推理
python seq_cls_infer.py --device gpu --model_dir ernie-3.0-medium-zh-afqmc
```
运行完成后返回的结果如下：

```bash
[INFO] fastdeploy/runtime.cc(469)::Init	Runtime initialized with Backend::ORT in Device::CPU.
Batch id:0, example id:0, sentence1:花呗收款额度限制, sentence2:收钱码，对花呗支付的金额有限制吗, label:1, similarity:0.5819
Batch id:1, example id:0, sentence1:花呗支持高铁票支付吗, sentence2:为什么友付宝不支持花呗付款, label:0, similarity:0.9979
```

### 参数说明

`seq_cls_infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。以下为各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定部署模型的目录， |
|--batch_size |最大可测的 batch size，默认为 1|
|--max_length |最大序列长度，默认为 128|
|--device | 运行的设备，可选范围: ['cpu', 'gpu']，默认为'cpu' |
|--backend | 支持的推理后端，可选范围: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，默认为'onnx_runtime' |
|--use_fp16 | 是否使用FP16模式进行推理。使用tensorrt和paddle_tensorrt后端时可开启，默认为False |
|--use_fast| 是否使用FastTokenizer加速分词阶段。默认为True|