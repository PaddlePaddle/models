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
# Download Deployment Example
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/python

#  Download LiteSeg Model and Test Image
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer.tgz
tar -xvf PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer.tgz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# CPU deployment
python infer.py --model PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer --image cityscapes_demo.png --device cpu
# GPU deployment
python infer.py --model PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer --image cityscapes_demo.png --device gpu
# TensorRT inference on GPU (note: if you run TensorRT inference the first time, there is a serialization of the model, which is time-consuming and requires patience)
python infer.py --model PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer --image cityscapes_demo.png --device gpu --use_trt True
```
The results of the completed visualisation are shown below:

Test Image:

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/203298832-f29fdcc1-a7f3-495a-8e39-fb67369292fb.png"  width = "50%" >
</div>

The Result after segmentation：
<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/203298024-4ed3b8ee-f393-4107-9e14-d52ad7bcbb89.png"  width = "50%" >
</div>

