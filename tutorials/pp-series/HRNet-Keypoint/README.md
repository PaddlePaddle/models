# [Deep High-Resolution Representation Learning for Human Pose Estimation (CVPR 2019)](https://arxiv.org/abs/1902.09212)

## 1 Introduction
This is the paddle code of [Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212).  
In this work, we are interested in the human pose estimation problem with a focus on learning reliable high-resolution representations. Most existing methods recover high-resolution representations from low-resolution representations produced by a high-to-low resolution network. Instead, our proposed network maintains high-resolution representations through the whole process. We start from a high-resolution subnetwork as the first stage, gradually add high-to-low resolution subnetworks one by one to form more stages, and connect the mutli-resolution subnetworks in parallel. We conduct repeated multi-scale fusions such that each of the high-to-low resolution representations receives information from other parallel representations over and over, leading to rich high-resolution representations. As a result, the predicted keypoint heatmap is potentially more accurate and spatially more precise. We empirically demonstrate the effectiveness of our network through the superior pose estimation results over two benchmark datasets: the COCO keypoint detection dataset and the MPII Human Pose dataset.

## 2 How to use

### 2.1 Environment

### Requirements:
- PaddlePaddle 2.2
- OS 64 bit
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64 bit
- pip/pip3(9.0.1+), 64 bit
- CUDA >= 10.1
- cuDNN >= 7.6

### Installation
#### 1. Install PaddlePaddle
```
# CUDA10.1
python -m pip install paddlepaddle-gpu==2.2.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

- For more CUDA version or environment to quick install, please refer to the [PaddlePaddle Quick Installation document](https://www.paddlepaddle.org.cn/install/quick)
- For more installation methods such as conda or compile with source code, please refer to the [installation document](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)

Please make sure that your PaddlePaddle is installed successfully and the version is not lower than the required version. Use the following command to verify.

```
# check
>>> import paddle
>>> paddle.utils.run_check()

# confirm the paddle's version
python -c "import paddle; print(paddle.__version__)"
```

**Note**

1.  If you want to train the model on multi-GPU, please install NCCL at first.

#### 2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
#### 3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
#### 4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── config
   ├── dataset
   ├── figures
   ├── lib
   ├── log
   ├── output
   ├── tools
   ├── README.md
   └── requirements.txt
   ```

### 2.2 Data preparation
#### COCO Data Download
- The coco dataset is downloaded automatically through the code. The dataset is large and takes a long time to download

    ```
    # automatically download coco datasets by executing code
    python dataset/download_coco.py
    ```

    after code execution, the organization structure of coco dataset file is：
    ```
    >>cd dataset
    >>tree
    ├── annotations
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   |   ...
    ├── train2017
    │   ├── 000000000009.jpg
    │   ├── 000000580008.jpg
    │   |   ...
    ├── val2017
    │   ├── 000000000139.jpg
    │   ├── 000000000285.jpg
    │   |   ...
    |   ...
    ```
- If the coco dataset has been downloaded  
    The files can be organized according to the above data file organization structure.

### 2.3 Training & Evaluation & Test

We provides scripts for training, evalution and test with various features according to different configure.

```bash
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/hrnet_w32_256x192.yml

# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/hrnet_w32_256x192.yml

# GPU evaluation
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/hrnet_w32_256x192.yml -o weights=https://paddle-model-ecology.bj.bcebos.com/model/hrnet_pose/hrnet_w32_256x192.pdparams

# test
python tools/infer.py -c configs/hrnet_w32_256x192.yml --infer_img=dataset/test_image/hrnet_demo.jpg -o weights=https://paddle-model-ecology.bj.bcebos.com/model/hrnet_pose/hrnet_w32_256x192.pdparams

# training with distillation
python tools/train.py -c configs/lite_hrnet_30_256x192_coco.yml  --distill_config=./configs/hrnet_w32_256x192_teacher.yml

# training with PACT quantization on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/lite_hrnet_30_256x192_coco_pact.yml

# training with PACT quantization on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/lite_hrnet_30_256x192_coco_pact.yml

# GPU evaluation with PACT quantization
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/lite_hrnet_30_256x192_coco_pact.yml -o weights=https://paddle-model-ecology.bj.bcebos.com/model/hrnet_pose/lite_hrnet_30_256x192_coco_pact.pdparams

# test with PACT quantization
python tools/infer.py -c configs/lite_hrnet_30_256x192_coco_pact.yml
--infer_img=dataset/test_image/hrnet_demo.jpg -o weights=https://paddle-model-ecology.bj.bcebos.com/model/hrnet_pose/lite_hrnet_30_256x192_coco.pdparams

```

### 2.4 Export model & Inference

```bash
# export model
python tools/export_model.py -c configs/hrnet_w32_256x192.yml -o  weights=https://paddle-model-ecology.bj.bcebos.com/model/hrnet_pose/hrnet_w32_256x192.pdparams

# inference
python deploy/infer.py  --model_dir=output_inference/hrnet_w32_256x192/ --image_file=dataset/test_image/hrnet_demo.jpg

# export model with lite model
python tools/export_model.py -c configs/lite_hrnet_30_256x192_coco.yml -o  weights=https://paddle-model-ecology.bj.bcebos.com/model/hrnet_pose/lite_hrnet_30_256x192_coco.pdparams

# inference with lite model
python deploy/infer.py  --model_dir=output_inference/lite_hrnet_30_256x192_coco/ --image_file=dataset/test_image/hrnet_demo.jpg

# export model with PACT quantization
python tools/export_model.py -c configs/lite_hrnet_30_256x192_coco_pact.yml -o  weights=https://paddle-model-ecology.bj.bcebos.com/model/hrnet_pose/lite_hrnet_30_256x192_coco_pact.pdparams

# inference with PACT quantization
python deploy/infer.py  --model_dir=output_inference/lite_hrnet_30_256x192_coco_pact/ --image_file=dataset/test_image/hrnet_demo.jpg

```

## 3 Result
COCO Dataset

| Model  | Input Size | AP(%, coco val) |   Model Download | Config File | Inference model size |
| :----------: | -------- | :----------: | :------------: | :---: | :---: |
| HRNet-w32             | 256x192  |     76.9     | [hrnet_w32_256x192.pdparams](https://paddle-model-ecology.bj.bcebos.com/model/hrnet_pose/hrnet_w32_256x192.pdparams) | [config](./configs/hrnet_w32_256x192.yml)           | 118M |
| LiteHRNet-30          | 256x192  |     69.4     | [lite_hrnet_30_256x192_coco.pdparams](https://paddle-model-ecology.bj.bcebos.com/model/hrnet_pose/lite_hrnet_30_256x192_coco.pdparams) | [config](./configs/lite_hrnet_30_256x192_coco.yml) | 26M |
| LiteHRNet-30-distillation    | 256x192  |     69.9     |[lite_hrnet_30_256x192_coco.pdparams](https://paddle-model-ecology.bj.bcebos.com/model/hrnet_pose/lite_hrnet_30_256x192_coco_dist.pdparams) | [config](./configs/lite_hrnet_30_256x192_coco_pact.yml)       | 26M |
| LiteHRNet-30-PACT         | 256x192  |     68.9     | [lite_hrnet_30_256x192_coco_pact.pdparams](https://paddle-model-ecology.bj.bcebos.com/model/hrnet_pose/lite_hrnet_30_256x192_coco_pact.pdparams) | [config](./configs/lite_hrnet_30_256x192_coco_pact.yml)         | 8.0M |


* note: Inference model size is obtained by summing `pdiparams` and `pdmodel` file size.


![](/dataset/test_image/hrnet_demo.jpg)
![](/deploy/output/hrnet_demo_vis.jpg)

## Citation
````
@inproceedings{cheng2020bottom,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}
````
