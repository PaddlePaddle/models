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
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/python

# 下载LiteSeg模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer.tgz
tar -xvf PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer.tgz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# CPU推理
python infer.py --model PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer --image cityscapes_demo.png --device cpu
# GPU推理
python infer.py --model PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer --image cityscapes_demo.png --device gpu
# GPU上使用TensorRT推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python infer.py --model PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer --image cityscapes_demo.png --device gpu --use_trt True
```
运行完成可视化结果如下图所示:

原始图像：

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/203298832-f29fdcc1-a7f3-495a-8e39-fb67369292fb.png"  width = "50%" >
</div>

分割后的图：
<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/203298024-4ed3b8ee-f393-4107-9e14-d52ad7bcbb89.png"  width = "50%" >
</div>

