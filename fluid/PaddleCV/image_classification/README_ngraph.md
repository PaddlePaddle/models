
# PaddlePaddle inference and training script
This directory contains configuration and instructions to run the PaddlePaddle + nGraph for a local training and inference.

# How to build PaddlePaddle framework with NGraph engine
In order to build the PaddlePaddle + nGraph engine and run proper scripti,  follow up a few steps:
1. Install PaddlePaddle project
2. set env exports for nGraph and OMP
3. run the inference/training script

Curently supported models:
* ResNet50 (inference and training).

Short description of aforementioned steps:

## 1. Install PaddlePaddle
Follow PaddlePaddle [installation instruction](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification#installation) to install PaddlePaddle. If you build PaddlePaddle yourself, please use the following cmake arguments and ensure to set -DWITH_NGRAPH=ON.  
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_DOC=OFF -DWITH_GPU=OFF -DWITH_DISTRIBUTE=OFF -DWITH_MKLDNN=ON -DWITH_MKL=ON -DWITH_GOLANG=OFF -DWITH_SWIG_PY=ON -DWITH_STYLE_CHECK=OFF -DWITH_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DWITH_PROFILER=OFF -DWITH_NGRAPH=ON
```

## 2. Set env exports for nGraph and OMP
Set the following exports needed for running nGraph:
```
export FLAGS_use_ngraph=true
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=<num_cpu_cores>
```

## 3. How the benchmark script might be run.
If everything built sucessfully, you can run command in ResNet50 nGraph session in script [run.sh](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleCV/image_classification/run.sh) to start the benchmark job locally. You will need to uncomment the `#ResNet50 nGraph` part of script.

Above is training job using the nGraph, to run the inference job using the nGraph:

Please download the pre-trained resnet50 model from [supported models](https://github.com/PaddlePaddle/models/tree/72dcc7c1a8d5de9d19fbd65b4143bd0d661eee2c/fluid/PaddleCV/image_classification#supported-models-and-performances). Run the inference script with use_gpu to be false.

```
python infer.py --use_gpu=False --model=ResNet50
```
