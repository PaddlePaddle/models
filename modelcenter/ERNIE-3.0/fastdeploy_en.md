## 0. FastDeploy

FastDeploy is an Easy-to-use and High Performance AI model deployment toolkit for Cloud, Mobile and Edge with out-of-the-box and unified experience, end-to-end optimization for over 150+ Text, Vision, Speech and Cross-modal AI models. FastDeploy Supports AI model deployment on
**X86 CPU、NVIDIA GPU、ARM CPU、XPU、NPU、IPU** etc. You can switch different inference backends and hardware with a single line of code.

Deploying AI model in 3 steps with FastDeploy: (1)Install FastDeploy SDK;  (2)Use FastDeploy's API to implement the deployment code;  (3) Deploy.

**Notes** : This document downloads FastDeploy examples to complete the high performance deployment experience; only X86 CPUs, NVIDIA GPUs are shown for reasoning and GPU environments are ready by default (e.g. CUDA >= 11.2, etc.), if you need to deploy AI model on other hardware or learn about FastDeploy's full capabilities, please refer to [FastDeploy GitHub](https://github.com/PaddlePaddle/FastDeploy).

## 1. Install FastDeploy SDK
```
pip install fastdeploy-gpu-python==0.0.0 -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html
```
## 2. Run Deployment Example
```
# download deployment example
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/text/ernie-3.0/python

#  download the fine-tuned ERNIE 3.0 model trained from the AFQMC dataset
wget https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-afqmc.tgz
tar xvfz ernie-3.0-medium-zh-afqmc.tgz

# CPU deployment
python seq_cls_infer.py --device cpu --model_dir ernie-3.0-medium-zh-afqmc

# GPU deployment
python seq_cls_infer.py --device gpu --model_dir ernie-3.0-medium-zh-afqmc
```
The results returned after the operation is completed are as follows:

```bash
[INFO] fastdeploy/runtime.cc(469)::Init	Runtime initialized with Backend::ORT in Device::CPU.
Batch id:0, example id:0, sentence1:花呗收款额度限制, sentence2:收钱码，对花呗支付的金额有限制吗, label:1, similarity:0.5819
Batch id:1, example id:0, sentence1:花呗支持高铁票支付吗, sentence2:为什么友付宝不支持花呗付款, label:0, similarity:0.9979
```

### Parameter Description

`seq_cls_infer.py` In addition to the command line parameters in the above example, more command line parameters are also supported. The following is a description of each command line parameter.

| Parameter |Parameter Description |
|----------|--------------|
|--model_dir | Specify the directory where the model is deployed， |
|--batch_size |Maximum measurable batch size，default 1|
|--max_length |Maximum sequence length，default 128|
|--device | equipment running，Optional range: ['cpu', 'gpu']，default'cpu' |
|--backend | Supported Inference Backends，Optional range: ['onnx_runtime', 'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']，default 'onnx_runtime' |
|--use_fp16 | Whether to use FP16 mode for inference。Use tensorrt and paddle_tensorrt can be turned on when backend，default False |
|--use_fast| Whether to use FastTokenizer to speed up the word segmentation stage。default True|