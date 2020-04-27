English | [中文](README.md)

# Image Classification and Model Zoo

> Note: The image classification library is migrated to new github repository: [https://github.com/PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas). More details about model zoo, high-end applications and many other new features can be seen in the new repository.

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
    - [Installation](#installation)
    - [Data preparation](#data-preparation)
    - [Training](#training)
    - [Finetuning](#finetuning)
    - [Evaluation](#evaluation)
    - [Inference](#inference)
- [Advanced Usage](#advanced-usage)
    - [Mixup Training](#mixup-training)
    - [Using Mixed-Precision Training](#using-mixed-precision-training)
    - [Profiling](#profiling)
    - [Preprocessing with Nvidia DALI](#preprocessing-with-nvidia-dali)
    - [Custom Dataset](#custom-dataset)
- [Supported Models and Performances](#supported-models-and-performances)
- [Reference](#reference)
- [Update](#update)
- [Contribute](#contribute)

---

## Introduction

Image classification, which is an important field of computer vision, is to classify images into pre-defined labels. Recently, many researchers have developed different kinds of neural networks and highly improved the classification performance. This page introduces how to do image classification with PaddlePaddle Fluid.

We also recommend users to take a look at the  [IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/122278)

## Quick Start

### Installation

Running samples in this directory requires Python 2.7 and later, CUDA 8.0 and later, CUDNN 7.0 and later, python package: numpy and opencv-python, PaddelPaddle Fluid v1.6 and later, the latest release version is recommended, If the PaddlePaddle on your device is lower than v1.6, please follow the instructions in [installation document](https://www.paddlepaddle.org.cn/install/quick) and make an update.

### Data preparation

An example for ImageNet classification is as follows.
For Linux system, preparation of imagenet data can be done as:

```bash
cd data/ILSVRC2012/
sh download_imagenet2012.sh
```

In the shell script ```download_imagenet2012.sh```,  there are three steps to prepare data:

**step-1:** Register at ```image-net.org``` first in order to get a pair of ```Username``` and ```AccessKey```, which are used to download ImageNet data.

**step-2:** Download ImageNet-2012 dataset from website. The training and validation data will be downloaded into folder "train" and "val" respectively. Please note that the size of data is more than 40 GB, it will take much time to download. Users who have downloaded the ImageNet data can organize it into ```data/ILSVRC2012``` directly.

**step-3:** Download training and validation label files. There are two label files which contain train and validation image labels respectively:

* train_list.txt: label file of imagenet-2012 training set, with each line seperated by ```SPACE```, like:
```
train/n02483708/n02483708_2436.jpeg 369
```
* val_list.txt: label file of imagenet-2012 validation set, with each line seperated by ```SPACE```, like.
```
val/ILSVRC2012_val_00000001.jpeg 65
```

Note: You may need to modify the data path in reader.py to load data correctly.
**For windows system, Users should download ImageNet data by themselves. and the label list can be downloaded in [Here](http://paddle-imagenet-models.bj.bcebos.com/ImageNet_label.tgz)**

### Training

After data preparation, one can start the training step by:

```
python train.py \
       --model=ResNet50 \
       --batch_size=256 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --lr=0.1
```
or running run.sh scripts

```bash
bash run.sh train model_name
```

**multiprocess training:**

If you have multiple gpus, this method is strongly recommended, because it can improve training speed dramatically.
You can start the multiprocess training step by:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch train.py \
       --model=ResNet50 \
       --batch_size=256 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --reader_thread=4 \
       --lr=0.1
```

or reference scripts/train/ResNet50_dist.sh

**parameter introduction:**

Environment settings:

* **data_dir**: the data root directory Default: "./data/ILSVRC2012".
* **model_save_dir**: the directory to save trained model. Default: "output".
* **pretrained_model**: load model path for pretraining. Default: None.
* **checkpoint**: load the checkpoint path to resume. Default: None.
* **print_step**: the batch steps interval to print log. Default: 10.
* **save_step**: the epoch steps interval to save checkpoints. Default:1.

Solver and hyperparameters:

* **model**: name model to use. Default: "ResNet50".
* **total_images**: total number of images in the training set. Default: 1281167.
* **class_dim**: the class number of the classification task. Default: 1000.
* **image_shape**: input size of the network. Default: 3 224 224 .
* **num_epochs**: the number of epochs. Default: 120.
* **batch_size**: the batch size of all devices. Default: 8.
* **test_batch_size**: the test batch size, Default: 16
* **lr_strategy**: learning rate changing strategy. Default: "piecewise_decay".
* **lr**: initialized learning rate. Default: 0.1.
* **l2_decay**: L2_decay parameter. Default: 1e-4.
* **momentum_rate**: momentum_rate. Default: 0.9.
* **step_epochs**: decay step of piecewise step, Default: [30,60,90].
* **decay_epochs**: decay epoch of exponential decay, Default: 2.4.
* **decay_rate**: decay rate of exponential decay, Default: 0.97.

Reader and preprocess:

* **lower_scale**: the lower scale in random crop data processing, upper is 1.0. Default:0.08.
* **lower_ratio**: the lower ratio in ramdom crop. Default:3./4. .
* **upper_ration**: the upper ratio in ramdom crop. Default:4./3. .
* **resize_short_size**: the resize_short_size. Default: 256.
* **use_mixup**: whether to use mixup data processing or not. Default:False.
* **mixup_alpha**: the mixup_alpha parameter. Default: 0.2.
* **use_aa**: whether to use auto augment data processing or not. Default:False.
* **reader_thread**: the number of threads in multi thread reader, Default: 8
* **reader_buf_size**: the buff size of multi thread reader, Default: 2048
* **interpolation**: interpolation method, Default: None
* **image_mean**: image mean, Default: [0.485, 0.456, 0.406]
* **image_std**: image std, Default: [0.229, 0.224, 0.225]


Switch:

* **validate**: whether to validate when training. Default: True.
* **use_gpu**: whether to use GPU or not. Default: True.
* **use_label_smoothing**: whether to use label_smoothing or not. Default:False.
* **label_smoothing_epsilon**: the label_smoothing_epsilon. Default:0.1.
* **padding_type**: padding type of convolution for efficientNet, Default: "SAME".
* **use_se**: whether to use Squeeze-and-Excitation module in efficientNet, Default: True.
* **use_ema**: whether to use ExponentialMovingAverage or not. Default: False.
* **ema_decay**: the value of ExponentialMovingAverage decay rate. Default: 0.9999.

Profiling:

* **enable_ce**: whether to start CE, Default: False
* **random_seed**: random seed, Default: None
* **is_profiler**: whether to start profilier, Default: 0
* **profilier_path**: path to save profilier output, Default: 'profilier_path'
* **max_iter**: maximum training batch, Default: 0
* **same_feed**: whether to feed same data in the net, Default: 0


**data reader introduction:** Data reader is defined in ```reader.py```, default reader is implemented by opencv. In the [Training](#training) Stage, random crop and flipping are applied, while center crop is applied in the [Evaluation](#evaluation) and [Inference](#inference) stages. Supported data augmentation includes:

* rotation
* color jitter
* random crop
* center crop
* resize
* flipping
* auto augment

### Finetuning

Finetuning is to finetune model weights in a specific task by loading pretrained weights. One can download [pretrained models](#supported-models-and-performances) and set its path to ```path_to_pretrain_model```, one can finetune a model by running following command:

```
python train.py \
       --model=model_name \
       --pretrained_model=${path_to_pretrain_model}
```

Note: Add and adjust other parameters accroding to specific models and tasks.

### Evaluation

Evaluation is to evaluate the performance of a trained model. One can download [pretrained models](#supported-models-and-performances) and set its path to ```path_to_pretrain_model```. Then top1/top5 accuracy can be obtained by running the following command:

**parameters**

* **save_json_path**: whether to save output, default: None

```
python eval.py \
       --model=model_name \
       --pretrained_model=${path_to_pretrain_model}
```

Note: Add and adjust other parameters accroding to specific models and tasks.

### ExponentialMovingAverage Evaluation

Note: if you train model with flag use_ema, and you want to evaluate your ExponentialMovingAverage model, you should clean your saved model first.

```
python ema_clean.py \
       --ema_model_dir=your_ema_model_dir \
       --cleaned_model_dir=your_cleaned_model_dir

python eval.py \
       --model=model_name \
       --pretrained_model=your_cleaned_model_dir
```

### Inference

**some Inference stage unique parameters**

* **save_inference**: whether to save binary model, Default: False
* **topk**: the number of sorted predicated labels to show, Default: 1
* **class_map_path**: readable label filepath, Default: "/utils/tools/readable_label.txt"
* **save_json_path**: whether to save output, Default: None
* **image_path**: whether to indicate the single image path to predict, Default: None

Inference is used to get prediction score or image features based on trained models. One can download [pretrained models](#supported-models-and-performances) and set its path to ```path_to_pretrain_model```. Run following command then obtain prediction score.

```bash
python infer.py \
       --model=model_name \
       --pretrained_model=${path_to_pretrain_model}
```

Note: Add and adjust other parameters accroding to specific models and tasks.

## Advanced Usage

### Mixup Training
Set --use_mixup=True to start Mixup training, all of the models with a suffix "_vd" is training by mixup.

Refer to [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

### Using Mixed-Precision Training

Set --use_fp16=True to sart Automatic Mixed Precision (AMP) Training. During the training process, the float16 data type will be used to speed up the training performance. You may need to use the --scale_loss parameter to avoid the accuracy dropping, such as setting --scale_loss=128.0.

After configuring the data path (modify the value of `DATA_DIR` in [scripts/train/ResNet50_fp16.sh](scripts/train/ResNet50_fp16.sh)), you can enable ResNet50 to start AMP Training by executing the command of `bash run.sh train ResNet50_fp16`.

Refer to [PaddlePaddle/Fleet](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/collective/resnet) for the multi-machine and multi-card training.

Performing on Tesla V100 single machine with 8 cards, two machines with 16 cards and four machines with 32 cards, the performance of ResNet50 AMP training is shown as below (enable DALI).

* BatchSize = 256

nodes*crads|throughput|speedup|test\_acc1|test\_acc5
---|---|---|---|---
1*1|1035 ins/s|1|0.75333|0.92702
1*8|7840 ins/s|7.57|0.75603|0.92771
2*8|14277 ins/s|13.79|0.75872|0.92793
4*8|28594 ins/s|27.63|0.75253|0.92713

* BatchSize = 128

nodes*crads|throughput|speedup|test\_acc1|test\_acc5
---|---|---|---|---
1*1|936  ins/s|1|0.75280|0.92531
1*8|7108 ins/s|7.59|0.75832|0.92771
2*8|12343 ins/s|13.18|0.75766|0.92723
4*8|24407 ins/s|26.07|0.75859|0.92871

### Preprocessing with Nvidia DALI

[Nvidia DALI](https://github.com/NVIDIA/DALI) can be used to preprocess input images, which could speed up training and achieve higher GPU utilization.

At present, DALI preprocessing supports the standard ImageNet pipeline (random crop -> resize -> flip -> normalize), it supports dataset in both file list or plain folder format.

DALI preprocessing can be enabled with the `--use_dali=True` command line flag.
For example, training ShuffleNet v2 0.25x with the following command should
reach a throughput of over 10000 images/second, and GPU utilization should be
above 85%.

``` bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_fraction_of_gpu_memory_to_use=0.80

python -m paddle.distributed.launch train.py \
       --model=ShuffleNetV2_x0_25 \
       --batch_size=2048 \
       --lr_strategy=cosine_decay_warmup \
       --num_epochs=240 \
       --lr=0.5 \
       --l2_decay=3e-5 \
       --lower_scale=0.64 \
       --lower_ratio=0.8 \
       --upper_ratio=1.2 \
       --use_dali=True

```

For more details please refer to [Documentation on DALI Paddle Plugin](https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/plugins/paddle_tutorials.html).

#### NOTES
1. PaddlePaddle with version 1.6 or above is required, and it must be compiled
with GCC 5.4 and up.
2. Nvidia DALI should include this PR [#1371](https://github.com/NVIDIA/DALI/pull/1371). Please refer to [this doc](https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/installation.html) and install nightly version or build from source.
3. Since DALI utilize the GPU for preprocessing, it will take up some GPU
   memory. Please reduce the memory used by paddle by setting the
   `FLAGS_fraction_of_gpu_memory_to_use` environment variable to a smaller
   number (e.g., `0.8`)

### Custom Dataset



## Supported Models and Performances

The image classification models currently supported by PaddlePaddle are listed in the table. It shows the top-1/top-5 accuracy on the ImageNet-2012 validation set of these models, the inference time of Paddle Fluid and Paddle TensorRT based on dynamic link library(test GPU model: Tesla P4).
Pretrained models can be downloaded by clicking related model names.

#### Note

- Some special settings

<table>
<tr>
    <td><b>Model</b>
    </td>
    <td><b>Resolution</b>
    </td>
    <td><b>Parameter: resize_short_size</b>
    </td>
</tr>
<tr>
   <td>Inception, Xception
   </td>
   <td>299
   </td>
   <td>320
   </td>
</tr>
<tr>
    <td> DarkNet53
    </td>
    <td>256
    </td>
    <td>256
    </td>
</tr>
<tr>
   <td>Fix_ResNeXt101_32x48d_wsl
   </td>
   <td>320
   </td>
   <td>320
   </td>
</tr>
<tr>
   <td rowspan="8"> EfficientNet: <br/><br/>
   In the inference phase, the resize_short_size increases 32 compared to the resolution <br/>
   and using the 2nd interpolation(cubic interpolation mode). <br/>
   The ExponentialMovingAverage method is also applied during the training process <br/>
   Please refer to <a href="https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/optimizer_cn.html#exponentialmovingaverage">ExponentialMovingAverage</a>
   </td>
   <td>B0: 224
   </td>
   <td>256
   </td>
</tr>
<tr>
   <td>B1: 240
   </td>
   <td>272
   </td>
</tr>
<tr>
   <td>B2: 260
   </td>
   <td>292
   </td>
</tr>
<tr>
   <td>B3: 300
   </td>
   <td>332
   </td>
</tr>
<tr>
   <td>B4: 380
   </td>
   <td>412
   </td>
</tr>
<tr>
   <td>B5: 456
   </td>
   <td>488
   </td>
</tr>
<tr>
   <td>B6: 528
   </td>
   <td>560
   </td>
</tr>
<tr>

   <td>B7: 600
   </td>
   <td>632
   </td>
</tr>
<tr>
    <td>Other models
    </td>
    <td>224
    </td>
    <td>256
    </td>
</tr>
</table>

- It's necessary to convert the train model to a binary model when appling dynamic link library to infer, One can do it by running following command:

    ```bash
    python infer.py\
        --model=model_name \
        --pretrained_model=${path_to_pretrained_model} \
        --save_inference=True
    ```

- The pretrained model of the ResNeXt101_wsl series network is converted from the pytorch model. Please refer to [RESNEXT WSL](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/) for details.

### AlexNet
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[AlexNet](http://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.tar) | 56.72% | 79.17% | 3.083 | 2.566 |

### SqueezeNet
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[SqueezeNet1_0](https://paddle-imagenet-models-name.bj.bcebos.com/SqueezeNet1_0_pretrained.tar) | 59.60% | 81.66% | 2.740 | 1.719 |
|[SqueezeNet1_1](https://paddle-imagenet-models-name.bj.bcebos.com/SqueezeNet1_1_pretrained.tar) | 60.08% | 81.85% | 2.751 | 1.282 |

### VGG Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[VGG11](https://paddle-imagenet-models-name.bj.bcebos.com/VGG11_pretrained.tar) | 69.28% | 89.09% | 8.223 | 6.619 |
|[VGG13](https://paddle-imagenet-models-name.bj.bcebos.com/VGG13_pretrained.tar) | 70.02% | 89.42% | 9.512 | 7.566 |
|[VGG16](https://paddle-imagenet-models-name.bj.bcebos.com/VGG16_pretrained.tar) | 72.00% | 90.69% | 11.315 | 8.985 |
|[VGG19](https://paddle-imagenet-models-name.bj.bcebos.com/VGG19_pretrained.tar) | 72.56% | 90.93% | 13.096 | 9.997 |

### MobileNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[MobileNetV1_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_25_pretrained.tar) | 51.43% | 75.46% | 2.283 | 0.838 |
|[MobileNetV1_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_5_pretrained.tar) | 63.52% | 84.73% | 2.378 | 1.052 |
|[MobileNetV1_x0_75](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_75_pretrained.tar) | 68.81% | 88.23% | 2.540 | 1.376 |
|[MobileNetV1](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) | 70.99% | 89.68% | 2.609 |1.615 |
|[MobileNetV2_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_25_pretrained.tar) | 53.21% | 76.52% | 4.267 | 2.791 |
|[MobileNetV2_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_5_pretrained.tar) | 65.03% | 85.72% | 4.514 | 3.008 |
|[MobileNetV2_x0_75](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_75_pretrained.tar) | 69.83% | 89.01% | 4.313 | 3.504 |
|[MobileNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) | 72.15% | 90.65% | 4.546 | 3.874 |
|[MobileNetV2_x1_5](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x1_5_pretrained.tar) | 74.12% | 91.67% | 5.235 | 4.771 |
|[MobileNetV2_x2_0](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x2_0_pretrained.tar) | 75.23% | 92.58% | 6.680 | 5.649 |
|[MobileNetV3_small_x1_0](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_pretrained.tar) | 67.46% | 87.12% | 6.809 |  |

### ShuffleNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[ShuffleNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar) | 68.80% | 88.45% | 6.101 | 3.616 |
|[ShuffleNetV2_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_25_pretrained.tar) | 49.90% | 73.79% | 5.956 | 2.505 |
|[ShuffleNetV2_x0_33](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_33_pretrained.tar) | 53.73% | 77.05% | 5.896 | 2.519 |
|[ShuffleNetV2_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_5_pretrained.tar) | 60.32% | 82.26% | 6.048 | 2.642 |
|[ShuffleNetV2_x1_5](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x1_5_pretrained.tar) | 71.63% | 90.15% | 6.113 | 3.164 |
|[ShuffleNetV2_x2_0](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x2_0_pretrained.tar) | 73.15% | 91.20% | 6.430 | 3.954 |
|[ShuffleNetV2_swish](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_swish_pretrained.tar) | 70.03% | 89.17% | 6.078 | 4.976 |

### AutoDL Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[DARTS_4M](https://paddle-imagenet-models-name.bj.bcebos.com/DARTS_GS_4M_pretrained.tar) | 75.23% | 92.15% | 13.572 | 6.335 |
|[DARTS_6M](https://paddle-imagenet-models-name.bj.bcebos.com/DARTS_GS_6M_pretrained.tar) | 76.03% | 92.79% | 16.406 | 6.864 |
- AutoDL is improved based on DARTS, Local Rademacher Complexity is introduced to control overfitting, and model size is flexibly adjusted through Resource Constraining.

### ResNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[ResNet18](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar) | 70.98% | 89.92% | 3.456 | 2.261 |
|[ResNet18_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_vd_pretrained.tar) | 72.26% | 90.80% | 3.847 | 2.404 |
|[ResNet34](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar) | 74.57% | 92.14% | 5.668 | 3.424 |
|[ResNet34_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_vd_pretrained.tar) | 75.98% | 92.98% | 6.089 | 3.544 |
|[ResNet50](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar) | 76.50% | 93.00% | 8.787 | 5.137 |
|[ResNet50_vc](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vc_pretrained.tar) |78.35% | 94.03% | 9.013 | 5.285 |
|[ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar) | 79.12% | 94.44% | 9.058 | 5.259 |
|[ResNet50_vd_v2](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_v2_pretrained.tar)<sup>[1](#trans1)</sup> | 79.84% | 94.93% | 9.058 | 5.259 |
|[ResNet101](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) | 77.56% | 93.64% | 15.447 | 8.473 |
|[ResNet101_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar) | 80.17% | 94.97% | 15.685 | 8.574 |
|[ResNet152](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_pretrained.tar) | 78.26% | 93.96% | 21.816 | 11.646 |
|[ResNet152_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_vd_pretrained.tar) | 80.59% | 95.30% | 22.041 | 11.858 |
|[ResNet200_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet200_vd_pretrained.tar) | 80.93% | 95.33% | 28.015 | 14.896 |

<a name="trans1">[1]</a> The pretrained model is distilled based on the pretrained model of ResNet50_vd. Users can directly load the pretrained model through the structure of ResNet50_vd.

### Res2Net Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[Res2Net50_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net50_26w_4s_pretrained.tar) | 79.33% | 94.57% | 10.731 | 8.274 |
|[Res2Net50_vd_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net50_vd_26w_4s_pretrained.tar) | 79.75% | 94.91% | 11.012 | 8.493 |
|[Res2Net50_14w_8s](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net50_14w_8s_pretrained.tar) | 79.46% | 94.70% | 16.937 | 10.205 |
|[Res2Net101_vd_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net101_vd_26w_4s_pretrained.tar) | 80.64% | 95.22% | 19.612 | 14.651 |
|[Res2Net200_vd_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net200_vd_26w_4s_pretrained.tar) | 81.21% | 95.71% | 35.809 | 26.479 |

### ResNeXt Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_32x4d_pretrained.tar) | 77.75% | 93.82% | 12.863 | 9.241 |
|[ResNeXt50_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_vd_32x4d_pretrained.tar) | 79.56% | 94.62% | 13.673 | 9.162 |
|[ResNeXt50_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_64x4d_pretrained.tar) | 78.43% | 94.13% | 28.162 | 15.935 |
|[ResNeXt50_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_vd_64x4d_pretrained.tar) | 80.12% | 94.86% | 20.888 | 15.938 |
|[ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x4d_pretrained.tar) | 78.65% | 94.19% | 24.154 | 17.661 |
|[ResNeXt101_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_vd_32x4d_pretrained.tar) | 80.33% | 95.12% | 24.701 | 17.249 |
|[ResNeXt101_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_64x4d_pretrained.tar) | 79.35% | 94.52% | 41.073 | 31.288 |
|[ResNeXt101_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_vd_64x4d_pretrained.tar) | 80.78% | 95.20% | 42.277 | 32.620 |
|[ResNeXt152_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_32x4d_pretrained.tar) | 78.98% | 94.33% | 37.007 | 26.981 |
|[ResNeXt152_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_vd_32x4d_pretrained.tar) | 80.72% | 95.20% | 35.783 | 26.081 |
|[ResNeXt152_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_64x4d_pretrained.tar) | 79.51% | 94.71% | 58.966 | 47.915 |
|[ResNeXt152_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_vd_64x4d_pretrained.tar) | 81.08% | 95.34% | 60.947 | 47.406 |

### DenseNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[DenseNet121](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet121_pretrained.tar) | 75.66% | 92.58% | 12.437 | 5.592 |
|[DenseNet161](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet161_pretrained.tar) | 78.57% | 94.14% | 27.717 | 12.254 |
|[DenseNet169](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet169_pretrained.tar) | 76.81% | 93.31% | 18.941 | 7.742 |
|[DenseNet201](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet201_pretrained.tar) | 77.63% | 93.66% | 26.583 | 10.066 |
|[DenseNet264](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet264_pretrained.tar) | 77.96% | 93.85% | 41.495 | 14.740 |

### DPN Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[DPN68](https://paddle-imagenet-models-name.bj.bcebos.com/DPN68_pretrained.tar) | 76.78% | 93.43% | 18.446 | 6.199 |
|[DPN92](https://paddle-imagenet-models-name.bj.bcebos.com/DPN92_pretrained.tar) | 79.85% | 94.80% | 25.748 | 21.029 |
|[DPN98](https://paddle-imagenet-models-name.bj.bcebos.com/DPN98_pretrained.tar) | 80.59% | 95.10% | 29.421 | 13.411 |
|[DPN107](https://paddle-imagenet-models-name.bj.bcebos.com/DPN107_pretrained.tar) | 80.89% | 95.32% | 41.071 | 18.885 |
|[DPN131](https://paddle-imagenet-models-name.bj.bcebos.com/DPN131_pretrained.tar) | 80.70% | 95.14% | 41.179 | 18.246 |

### SENet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[SE_ResNet18_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet18_vd_pretrained.tar) | 73.33% | 91.38% | 4.715 | 3.061 |
|[SE_ResNet34_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet34_vd_pretrained.tar) | 76.51% | 93.20% | 7.475 | 4.299 |
|[SE_ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet50_vd_pretrained.tar) | 79.52% | 94.75% | 10.345 | 7.631 |
|[SE_ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt50_32x4d_pretrained.tar) | 78.44% | 93.96% | 14.916 | 12.305 |
|[SE_ResNeXt50_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt50_vd_32x4d_pretrained.tar) | 80.24% | 94.89% | 15.155 | 12.687 |
|[SE_ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt101_32x4d_pretrained.tar) | 79.12% | 94.20% | 30.085 | 23.218 |
|[SENet154_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SENet154_vd_pretrained.tar) | 81.40% | 95.48% | 71.892 | 53.131 |

### SENet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[SE_ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet50_vd_pretrained.tar) | 79.52% | 94.75% | 10.345 | 7.631 |
|[SE_ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt50_32x4d_pretrained.tar) | 78.44% | 93.96% | 14.916 | 12.305 |
|[SE_ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt101_32x4d_pretrained.tar) | 79.12% | 94.20% | 30.085 | 23.218 |
|[SENet154_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SENet154_vd_pretrained.tar) | 81.40% | 95.48% | 71.892 | 53.131 |

### Inception Series
| Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[GoogLeNet](https://paddle-imagenet-models-name.bj.bcebos.com/GoogLeNet_pretrained.tar) | 70.70% | 89.66% | 6.528 | 2.919 |
|[Xception41](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_pretrained.tar) | 79.30% | 94.53% | 13.757 | 7.885 |
|[Xception41_deeplab](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_deeplab_pretrained.tar) | 79.55% | 94.38% | 14.268 | 7.257 |
|[Xception65](https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_pretrained.tar) | 81.00% | 95.49% | 19.216 | 10.742 |
|[Xception65_deeplab](https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_deeplab_pretrained.tar) | 80.32% | 94.49% | 19.536 | 10.713 |
|[Xception71](https://paddle-imagenet-models-name.bj.bcebos.com/Xception71_pretrained.tar) | 81.11% | 95.45% | 23.291 | 12.154 |
|[InceptionV4](https://paddle-imagenet-models-name.bj.bcebos.com/InceptionV4_pretrained.tar) | 80.77% | 95.26% | 32.413 | 17.728 |

### DarkNet
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[DarkNet53](https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar) | 78.04% | 94.05% | 11.969 | 6.300 |

### ResNeXt101_wsl Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[ResNeXt101_32x8d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x8d_wsl_pretrained.tar) | 82.55% | 96.74% | 33.310 | 27.628 |
|[ResNeXt101_32x16d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x16d_wsl_pretrained.tar) | 84.24% | 97.26% | 54.320 | 47.599 |
|[ResNeXt101_32x32d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x32d_wsl_pretrained.tar) | 84.97% | 97.59% | 97.734 | 81.660 |
|[ResNeXt101_32x48d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x48d_wsl_pretrained.tar) | 85.37% | 97.69% | 161.722 |  |
|[Fix_ResNeXt101_32x48d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/Fix_ResNeXt101_32x48d_wsl_pretrained.tar) | 86.26% | 97.97% | 236.091 |  |

### EfficientNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[EfficientNetB0](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB0_pretrained.tar) | 77.38% | 93.31% | 10.303 | 4.334 |
|[EfficientNetB1](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB1_pretrained.tar)<sup>[2](#trans2)</sup> | 79.15% | 94.41% | 15.626 | 6.502 |
|[EfficientNetB2](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB2_pretrained.tar)<sup>[2](#trans2)</sup> | 79.85% | 94.74% | 17.847 | 7.558 |
|[EfficientNetB3](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB3_pretrained.tar)<sup>[2](#trans2)</sup> | 81.15% | 95.41% | 25.993 | 10.937 |
|[EfficientNetB4](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB4_pretrained.tar)<sup>[2](#trans2)</sup> | 82.85% | 96.23% | 47.734 | 18.536 |
|[EfficientNetB5](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB5_pretrained.tar)<sup>[2](#trans2)</sup> | 83.62% | 96.72% | 88.578 | 32.102 |
|[EfficientNetB6](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB6_pretrained.tar)<sup>[2](#trans2)</sup> | 84.00% | 96.88% | 138.670 | 51.059 |
|[EfficientNetB7](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB7_pretrained.tar)<sup>[2](#trans2)</sup> | 84.30% | 96.89% | 234.364 | 82.107 |
|[EfficientNetB0_small](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB0_Small_pretrained.tar)<sup>[3](#trans3)</sup> | 75.80% | 92.58% | 3.342 | 2.729 |

<a name="trans2">[2]</a> means the pretrained weight is converted form [original repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

<a name="trans3">[3]</a> means the pretrained weight is based on EfficientNetB0, removed Squeeze-and-Excitation module and use general convolution. This model speed is much faster.

### HRNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[HRNet_W18_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W18_C_pretrained.tar) | 76.92% | 93.39% | 23.013 | 11.601 |
|[HRNet_W30_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W30_C_pretrained.tar) | 78.04% | 94.02% | 25.793 | 14.367 |
|[HRNet_W32_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W32_C_pretrained.tar) | 78.28% | 94.24% | 29.564 | 14.328 |
|[HRNet_W40_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W40_C_pretrained.tar) | 78.77% | 94.47% | 33.880 | 17.616 |
|[HRNet_W44_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W44_C_pretrained.tar) | 79.00% | 94.51% | 36.021 | 18.990 |
|[HRNet_W48_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W48_C_pretrained.tar) | 78.95% | 94.42% | 30.064 | 19.963 |
|[HRNet_W64_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W64_C_pretrained.tar) | 79.30% | 94.61% | 38.921 | 24.742 |

### ResNet_ACNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[ResNet50_ACNet](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_ACNet_pretrained.tar)<sub>1</sub> | 76.71% | 93.24% | 13.205 | 8.804 |
|[ResNet50_ACNet](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_ACNet_deploy_pretrained.tar)<sub>2</sub> | 76.71% | 93.24% | 7.418 | 5.950 |

* Note:
    * `1`. deploy mode is set as False to eval.
    * `2`. Use `sh ./utils/acnet/convert_model.sh` to convert to trained model, and set deploy mode as True to eval.
    * `./utils/acnet/convert_model.sh` contains 4 parmeters, which are model name, input model directory, output model directory and class number.


## FAQ

**Q:** How to solve this problem when I try to train a 6-classes dataset with indicating pretrained_model parameter ?
```
Enforce failed. Expected x_dims[1] == labels_dims[1], but received x_dims[1]:1000 != labels_dims[1]:6.
```

**A:** It may be caused by dismatch dimensions. Please remove fc parameter in pretrained models, It usually named with a prefix ```fc_```

## Reference


- AlexNet: [imagenet-classification-with-deep-convolutional-neural-networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- ResNet: [Deep Residual Learning for Image Recognitio](https://arxiv.org/abs/1512.03385), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- ResNeXt: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431), Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
- SeResNeXt: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)Jie Hu, Li Shen, Samuel Albanie
- ShuffleNetV1: [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083), Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
- ShuffleNetV2: [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164), Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
- MobileNetV1: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861), Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
- MobileNetV2: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381v4.pdf), Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
- MobileNetV3: [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf), Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
- VGG: [Very Deep Convolutional Networks for Large-scale Image Recognition](https://arxiv.org/pdf/1409.1556), Karen Simonyan, Andrew Zisserman
- GoogLeNet: [Going Deeper with Convolutions](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf), Christian Szegedy1, Wei Liu2, Yangqing Jia
- Xception: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357), Franc ̧ois Chollet
- InceptionV4: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261), Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
- DarkNet: [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), Joseph Redmon, Ali Farhadi
- DenseNet: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993), Gao Huang, Zhuang Liu, Laurens van der Maaten
- DPN: [Dual Path Networks](https://arxiv.org/pdf/1707.01629.pdf), Yunpeng Chen, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, Jiashi Feng
- SqueezeNet: [SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE](https://arxiv.org/abs/1602.07360), Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
- ResNeXt101_wsl: [Exploring the Limits of Weakly Supervised Pretraining](https://arxiv.org/abs/1805.00932), Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, Laurens van der Maaten
- Fix_ResNeXt101_wsl: [Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423), Hugo Touvron, Andrea Vedaldi, Matthijs Douze, Herve ́ Je ́gou
- EfficientNet: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946), Mingxing Tan, Quoc V. Le
- Res2Net: [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169), Shang-Hua Gao, Ming-Ming Cheng, Kai Zhao, Xin-Yu Zhang, Ming-Hsuan Yang, Philip Torr
- HRNet: [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919), Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao
- DARTS: [DARTS: Differentiable Architecture Search](https://arxiv.org/pdf/1806.09055.pdf), Hanxiao Liu, Karen Simonyan, Yiming Yang
- ACNet: [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](https://arxiv.org/abs/1908.03930), Xiaohan Ding, Yuchen Guo, Guiguang Ding, Jungong Han



## Update
- 2018/12/03 **Stage1**: Update AlexNet, ResNet50, ResNet101, MobileNetV1
- 2018/12/23 **Stage2**: Update VGG Series, SeResNeXt50_32x4d, SeResNeXt101_32x4d, ResNet152
- 2019/01/31 Update MobileNetV2_x1_0
- 2019/04/01 **Stage3**: Update ResNet18, ResNet34, GoogLeNet, ShuffleNetV2
- 2019/06/12 **Stage4**:Update ResNet50_vc, ResNet50_vd, ResNet101_vd, ResNet152_vd, ResNet200_vd, SE154_vd InceptionV4, ResNeXt101_64x4d, ResNeXt101_vd_64x4d
- 2019/06/22 Update ResNet50_vd_v2
- 2019/07/02 **Stage5**: Update MobileNetV2_x0_5, ResNeXt50_32x4d, ResNeXt50_64x4d, Xception41, ResNet101_vd
- 2019/07/19 **Stage6**: Update ShuffleNetV2_x0_25, ShuffleNetV2_x0_33, ShuffleNetV2_x0_5, ShuffleNetV2_x1_0, ShuffleNetV2_x1_5, ShuffleNetV2_x2_0, MobileNetV2_x0_25, MobileNetV2_x1_5, MobileNetV2_x2_0, ResNeXt50_vd_64x4d, ResNeXt101_32x4d, ResNeXt152_32x4d
- 2019/08/01 **Stage7**: Update DarkNet53, DenseNet121. Densenet161, DenseNet169, DenseNet201, DenseNet264, SqueezeNet1_0, SqueezeNet1_1, ResNeXt50_vd_32x4d, ResNeXt152_64x4d, ResNeXt101_32x8d_wsl, ResNeXt101_32x16d_wsl, ResNeXt101_32x32d_wsl, ResNeXt101_32x48d_wsl, Fix_ResNeXt101_32x48d_wsl
- 2019/09/11 **Stage8**: Update ResNet18_vd，ResNet34_vd，MobileNetV1_x0_25，MobileNetV1_x0_5，MobileNetV1_x0_75，MobileNetV2_x0_75，MobilenNetV3_small_x1_0，DPN68，DPN92，DPN98，DPN107，DPN131，ResNeXt101_vd_32x4d，ResNeXt152_vd_64x4d，Xception65，Xception71，Xception41_deeplab，Xception65_deeplab，SE_ResNet50_vd
- 2019/09/20 Update EfficientNet
- 2019/11/28 **Stage9**: Update SE_ResNet18_vd，SE_ResNet34_vd，SE_ResNeXt50_vd_32x4d，ResNeXt152_vd_32x4d，Res2Net50_26w_4s，Res2Net50_14w_8s，Res2Net50_vd_26w_4s，HRNet_W18_C，HRNet_W30_C，HRNet_W32_C，HRNet_W40_C，HRNet_W44_C，HRNet_W48_C，HRNet_W64_C
- 2020/01/07 **Stage10**: Update AutoDL Series
- 2020/01/09 **Stage11**: Update Res2Net101_vd_26w_4s, Res2Net200_vd_26w_4s

## Contribute

If you can fix an issue or add a new feature, please open a PR to us. If your PR is accepted, you can get scores according to the quality and difficulty of your PR(0~5), while you got 10 scores, you can contact us for interview or recommendation letter.
