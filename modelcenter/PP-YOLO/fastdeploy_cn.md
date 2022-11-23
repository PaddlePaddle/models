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
cd FastDeploy/examples/vision/detection/paddledetection/python/

#下载YOLO模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyolo_r50vd_dcn_1x_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyolo_r50vd_dcn_1x_coco.tgz

# CPU推理
python infer_ppyolo.py --model_dir ppyolo_r50vd_dcn_1x_coco --image 000000014439.jpg --device cpu
# GPU推理
python infer_ppyolo.py --model_dir ppyolo_r50vd_dcn_1x_coco --image 000000014439.jpg --device gpu
# GPU上使用TensorRT推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python infer_ppyolo.py --model_dir ppyolo_r50vd_dcn_1x_coco --image 000000014439.jpg --device gpu --use_trt True
```