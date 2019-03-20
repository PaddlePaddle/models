
# PaddlePaddle inference and training script
This directory contains model configuration and tool used to run the PaddlePaddle + NGraph for a local training and inference.

# How to build PaddlePaddle framework with NGraph engine
In order to build the PaddlePaddle + NGraph engine and run proper script follow up a few steps:
1. build the PaddlePaddle project
2. download pre-trained model data
3. set env exports for nGraph and OMP
5. run the inference/training script

Curently supported models:
* ResNet50 (inference and training).

Short description of aforementioned steps:

## 1. Build paddle
Do it as you usually do. In case you never did it, here are instructions:
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_DOC=OFF -DWITH_GPU=OFF -DWITH_DISTRIBUTE=OFF -DWITH_MKLDNN=ON -DWITH_MKL=ON -DWITH_GOLANG=OFF -DWITH_SWIG_PY=ON -DWITH_STYLE_CHECK=OFF -DWITH_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DWITH_PROFILER=OFF -DWITH_NGRAPH=ON
```
## 2. Download pre-trained model:
In order to download model, go to data/ILSVRC2012/ directory and run download_resnet50.sh script:
```
$ cd data/ILSVRC2012/
$ ./download_resnet50.sh
```

## 3. Set env exports for nGraph and OMP
Set the following exports needed for running nGraph:
```
export FLAGS_use_ngraph=true
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=<num_cpu_cores>
```

## 4. How the benchmark script might be run.
If everything built sucessfully, you can run the following command to start the benchmark job locally or uncomment the `#ResNet50 ngraph` part of script.

Run the training job using the nGraph:
```
#ResNet50 nGraph:
numactl -l python train.py \
            --model=ResNet50 \
            --batch_size=256 \
            --total_images=1281167 \
            --class_dim=1000 \
            --image_shape=3,224,224 \
            --lr_strategy=none \
            --lr=0.001 \
            --num_epochs=120 \
            --with_mem_opt=False \
            --model_category=models_name \
            --model_save_dir=output/ \
            --use_gpu=False·

```
Run the inference job using the nGraph:
```
numactl -l python infer.py --use_gpu false --model=ResNet50
```
