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
cd FastDeploy/examples/vision/matting/ppmatting/python

#  Download PP-Matting Model and Test Image
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-512.tgz
tar -xvf PP-Matting-512.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_bgr.jpg
# CPU deployment
python infer.py --model PP-Matting-512 --image matting_input.jpg --bg matting_bgr.jpg --device cpu
# GPU deployment
python infer.py --model PP-Matting-512 --image matting_input.jpg --bg matting_bgr.jpg --device gpu
# TensorRT inference on GPU (note: if you run TensorRT inference the first time, there is a serialization of the model, which is time-consuming and requires patience)
python infer.py --model PP-Matting-512 --image matting_input.jpg --bg matting_bgr.jpg --device gpu --use_trt True
```

The results of the completed visualisation are shown below:
<div width="840">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852040-759da522-fca4-4786-9205-88c622cd4a39.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852587-48895efc-d24a-43c9-aeec-d7b0362ab2b9.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852116-cf91445b-3a67-45d9-a675-c69fe77c383a.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852554-6960659f-4fd7-4506-b33b-54e1a9dd89bf.jpg">
</div>