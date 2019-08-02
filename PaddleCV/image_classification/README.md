# Image Classification and Model Zoo

---
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
    - [Using Mixed-Precision Training](#using-mixed-precision-training)
    - [CE](#ce)
- [Supported Models and Performances](#supported-models-and-performances)
- [Reference](#reference)
- [Update](#update)
- [Contribute](#contribute)

## Introduction

Image classification, which is an important field of computer vision, is to classify an image into pre-defined labels. Recently, many researchers developed different kinds of neural networks and highly improve the classification performance. This page introduces how to do image classification with PaddlePaddle Fluid.

## Quick Start

### Installation

Running sample code in this directory requires Python 2.7 and later, PaddelPaddle Fluid v1.5 and later, the latest release version is recommended, If the PaddlePaddle on your device is lower than v1.5, please follow the instructions in [installation document](http://paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html) and make an update.

### Data preparation

An example for ImageNet classification is as follows. First of all, preparation of imagenet data can be done as:
```
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

You may need to modify the path in reader.py to load data correctly.

### Training

After data preparation, one can start the training step by:

```
python train.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --with_mem_opt=False \
       --with_inplace=True \
       --lr_strategy=piecewise_decay \
       --lr=0.1
```
**parameter introduction:**

* **model**: name model to use. Default: "SE_ResNeXt50_32x4d".
* **num_epochs**: the number of epochs. Default: 120.
* **batch_size**: the size of each mini-batch. Default: 256.
* **use_gpu**: whether to use GPU or not. Default: True.
* **total_images**: total number of images in the training set. Default: 1281167.
* **class_dim**: the class number of the classification task. Default: 1000.
* **image_shape**: input size of the network. Default: "3,224,224".
* **model_save_dir**: the directory to save trained model. Default: "output".
* **with_mem_opt**: whether to use memory optimization or not. Default: False.
* **with_inplace**: whether to use inplace memory optimization or not. Default: True.
* **lr_strategy**: learning rate changing strategy. Default: "piecewise_decay".
* **lr**: initialized learning rate. Default: 0.1.
* **pretrained_model**: model path for pretraining. Default: None.
* **checkpoint**: the checkpoint path to resume. Default: None.
* **data_dir**: the data path. Default: "./data/ILSVRC2012".
* **fp16**: whether to enable half precision training with fp16. Default: False.
* **scale_loss**: scale loss for fp16. Default: 1.0.
* **l2_decay**: L2_decay parameter. Default: 1e-4.
* **momentum_rate**: momentum_rate. Default: 0.9.
* **use_label_smoothing**: whether to use label_smoothing or not. Default:False.
* **label_smoothing_epsilon**: the label_smoothing_epsilon. Default:0.2.
* **lower_scale**: the lower scale in random crop data processing, upper is 1.0. Default:0.08.
* **lower_ratio**: the lower ratio in ramdom crop. Default:3./4. .
* **upper_ration**: the upper ratio in ramdom crop. Default:4./3. .
* **resize_short_size**: the resize_short_size. Default: 256.
* **use_mixup**: whether to use mixup data processing or not. Default:False.
* **mixup_alpha**: the mixup_alpha parameter. Default: 0.2.
* **is_distill**: whether to use distill or not. Default: False.

Or can start the training step by running the ```run.sh```.

**data reader introduction:** Data reader is defined in PIL: ```reader.py```and opencv: ```reader_cv2.py```, default reader is implemented by opencv. In [Training](#training), random crop and flipping are used, while center crop is used in [Evaluation](#evaluation) and [Inference](#inference) stages. Supported data augmentation includes:

* rotation
* color jitter (haven't implemented in cv2_reader)
* random crop
* center crop
* resize
* flipping

### Finetuning

Finetuning is to finetune model weights in a specific task by loading pretrained weights. One can download [pretrained models](#supported-models-and-performances) and set its path to ```path_to_pretrain_model```, one can finetune a model by running following command:

```
python train.py \
       --pretrained_model=${path_to_pretrain_model}
```

Note: Add and adjust other parameters accroding to specific models and tasks.

### Evaluation

Evaluation is to evaluate the performance of a trained model. One can download [pretrained models](#supported-models-and-performances) and set its path to ```path_to_pretrain_model```. Then top1/top5 accuracy can be obtained by running the following command:

```
python eval.py \
       --pretrained_model=${path_to_pretrain_model}
```

Note: Add and adjust other parameters accroding to specific models and tasks.

### Inference

Inference is used to get prediction score or image features based on trained models. One can download [pretrained models](#supported-models-and-performances) and set its path to ```path_to_pretrain_model```. Run following command then obtain prediction score.

```
python infer.py \
       --pretrained_model=${path_to_pretrain_model}
```

Note: Add and adjust other parameters accroding to specific models and tasks.

## Advanced Usage

### Using Mixed-Precision Training

You may add `--fp16=1` to start train using mixed precisioin training, which the training process will use float16 and the output model ("master" parameters) is saved as float32. You also may need to pass `--scale_loss` to overcome accuracy issues, usually `--scale_loss=8.0` will do.

Note that currently `--fp16` can not use together with `--with_mem_opt`, so pass `--with_mem_opt=0` to disable memory optimization pass.

### CE

CE is only for internal testing, don't have to set it.

## Supported Models and Performances

The image classification models currently supported by PaddlePaddle are listed in the table. It shows the top-1/top-5 accuracy on the ImageNet-2012 validation set of these models, the inference time of Paddle Fluid and Paddle TensorRT based on dynamic link library(test GPU model: Tesla P4). 
As the activation function ```swish``` and ```relu6``` which separately used in ShuffleNetV2_swish and MobileNetV2 net are not supported by Paddle TensorRT, inference acceleration performance of them doesn't significient improve. Pretrained models can be downloaded by clicking related model names.

- Note1: ResNet50_vd_v2 is the distilled version of ResNet50_vd. 
- Note2: The image resolution feeded in InceptionV4 and Xception net is ```299x299```, Fix_ResNeXt101_32x48d_wsl is ```320x320```, DarkNet is ```256x256```, others are ```224x224```.In test time, the resize_short_size of the DarkNet53 and Fix_ResNeXt101_32x48d_wsl series networks is the same as the width or height of the input image resolution, the InceptionV4 and Xception network resize_short_size is 320, and the other networks resize_short_size are 256.
- Note3: It's necessary to convert the train model to a binary model when appling dynamic link library to infer, One can do it by running following command:

    ```python infer.py --save_inference=True```
- Note4: The pretrained model of the ResNeXt101_wsl series network is converted from the pytorch model. Please go to [RESNEXT WSL](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/) for details.
    
### AlexNet
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[AlexNet](http://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.tar) | 56.72%/79.17% | 3.083 | 2.728 |

### SqueezeNet
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[SqueezeNet1_0](https://paddle-imagenet-models-name.bj.bcebos.com/SqueezeNet1_0_pretrained.tar) | 59.60%/81.66% | 2.740 | 1.688 |
|[SqueezeNet1_1](https://paddle-imagenet-models-name.bj.bcebos.com/SqueezeNet1_1_pretrained.tar) | 60.08%/81.85% | 2.751 | 1.270 |

### VGG
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[VGG11](https://paddle-imagenet-models-name.bj.bcebos.com/VGG11_pretrained.tar) | 69.28%/89.09% | 8.223 | 6.821 |
|[VGG13](https://paddle-imagenet-models-name.bj.bcebos.com/VGG13_pretrained.tar) | 70.02%/89.42% | 9.512 | 7.783 |
|[VGG16](https://paddle-imagenet-models-name.bj.bcebos.com/VGG16_pretrained.tar) | 72.00%/90.69% | 11.315 | 9.067 |
|[VGG19](https://paddle-imagenet-models-name.bj.bcebos.com/VGG19_pretrained.tar) | 72.56%/90.93% | 13.096 | 10.388 |

### MobileNet
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[MobileNetV1](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) | 70.99%/89.68% | 2.609 |1.615 |
|[MobileNetV2_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_25_pretrained.tar) | 53.21%/76.52% | 4.267 | 3.777 |
|[MobileNetV2_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_5_pretrained.tar) | 65.03%/85.72% | 4.514 | 4.150 |
|[MobileNetV2_x1_0](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) | 72.15%/90.65% | 4.546 | 5.278 |
|[MobileNetV2_x1_5](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x1_5_pretrained.tar) | 74.12%/91.67% | 5.235 | 6.909 |
|[MobileNetV2_x2_0](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x2_0_pretrained.tar) | 75.23%/92.58% | 6.680 | 7.658 |

### ShuffleNet
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[ShuffleNetV2_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_25_pretrained.tar) | 49.90%/73.79% | 5.956 | 2.961 |
|[ShuffleNetV2_x0_33](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_33_pretrained.tar) | 53.73%/77.05% | 5.896 | 2.941 |
|[ShuffleNetV2_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_5_pretrained.tar) | 60.32%/82.26% | 6.048 | 3.088 |
|[ShuffleNetV2_x1_0](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x1_0_pretrained.tar) | 68.80%/88.45% | 6.101 | 3.616 |
|[ShuffleNetV2_x1_5](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x1_5_pretrained.tar) | 71.63%/90.15% | 6.113 | 3.699 |
|[ShuffleNetV2_x2_0](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x2_0_pretrained.tar) | 73.15%/91.20% | 6.430 | 4.553 |
|[ShuffleNetV2_x1_0_swish](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar) | 70.03%/89.17% | 6.078 | 6.282 |

### ResNet
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[ResNet18](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar) | 70.98%/89.92% | 3.456 | 2.484 |
|[ResNet34](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar) | 74.57%/92.14% | 5.668 | 3.767 |
|[ResNet50](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar) | 76.50%/93.00% | 8.787 | 5.434 |
|[ResNet50_vc](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vc_pretrained.tar) |78.35%/94.03% | 9.013 | 5.463 |
|[ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar) | 79.12%/94.44% | 9.058 | 5.510 |
|[ResNet50_vd_v2](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_v2_pretrained.tar) | 79.84%/94.93% | 9.058 | 5.510 |
|[ResNet101](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) | 77.56%/93.64% | 15.447 | 8.779 |
|[ResNet101_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar) | 80.17%/94.97% | 15.685 | 8.878 |
|[ResNet152](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_pretrained.tar) | 78.26%/93.96% | 21.816 | 12.148 |
|[ResNet152_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_vd_pretrained.tar) | 80.59%/95.30% | 22.041 | 12.259 |
|[ResNet200_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet200_vd_pretrained.tar) | 80.93%/95.33% | 28.015 | 15.278 |

### ResNeXt
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_32x4d_pretrained.tar) | 77.75%/93.82% | 12.863 | 9.837 |
|[ResNeXt50_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_vd_32x4d_pretrained.tar) | 79.56%/94.62% | 13.673 | 9.991 |
|[ResNeXt50_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_64x4d_pretrained.tar) | 78.43%/94.13% | 28.162 | 18.271 |
|[ResNeXt50_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_vd_64x4d_pretrained.tar) | 80.12%/94.86% | 20.888 | 17.687 |
|[ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x4d_pretrained.tar) | 78.65%/94.19% | 24.154 | 21.387 |
|[ResNeXt101_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_64x4d_pretrained.tar) | 78.43%/94.13% | 41.073 | 38.736 |
|[ResNeXt101_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_vd_64x4d_pretrained.tar) | 80.78%/95.20% | 42.277 | 40.929 |
|[ResNeXt152_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_32x4d_pretrained.tar) | 78.98%/94.33% | 37.007 | 31.301 |
|[ResNeXt152_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_64x4d_pretrained.tar) | 79.51%/94.71% | 58.966 | 57.267 |

### DenseNet
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[DenseNet121](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet121_pretrained.tar) | 75.66%/92.58% | 12.437 | 5.813 |
|[DenseNet161](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet161_pretrained.tar) | 78.57%/94.14% | 27.717 | 12.861 |
|[DenseNet169](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet169_pretrained.tar) | 76.81%/93.31% | 18.941 | 8.146 |
|[DenseNet201](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet201_pretrained.tar) | 77.63%/93.66% | 26.583 | 10.549 |
|[DenseNet264](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet264_pretrained.tar) | 77.96%/93.85% | 41.495 | 15.574 |

### SENet
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[SE_ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt50_32x4d_pretrained.tar) | 78.44%/93.96% | 14.916 | 12.126 |
|[SE_ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt101_32x4d_pretrained.tar) | 79.12%/94.20% | 30.085 | 24.110 |
|[SENet154_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SE154_vd_pretrained.tar) | 81.40%/95.48% | 71.892 | 64.855 |

### Inception
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[GoogLeNet](https://paddle-imagenet-models-name.bj.bcebos.com/GoogleNet_pretrained.tar) | 70.70%/89.66% | 6.528 | 3.076 |
|[Xception_41](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_pretrained.tar) | 79.30%/94.53% | 13.757 | 10.831 |
|[InceptionV4](https://paddle-imagenet-models-name.bj.bcebos.com/InceptionV4_pretrained.tar) | 80.77%/95.26% | 32.413 | 18.154 |

### DarkNet
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[DarkNet53](https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar) | 78.04%/94.05% | 11.969 | 7.153 |

### ResNeXt101_wsl
|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[ResNeXt101_32x8d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x8d_wsl_pretrained.tar) | 82.55%/96.74% | 33.310 | 27.648 |
|[ResNeXt101_32x16d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x16d_wsl_pretrained.tar) | 84.24%/97.26% | 54.320 | 46.064 |
|[ResNeXt101_32x32d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x32d_wsl_pretrained.tar) | 84.97%/97.59% | 97.734 | 87.961 |
|[ResNeXt101_32x48d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x48d_wsl_pretrained.tar) | 85.37%/97.69% | 161.722 |  |
|[Fix_ResNeXt101_32x48d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/Fix_ResNeXt101_32x48d_wsl_pretrained.tar) | 86.26%/97.97% | 236.091 |  |

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
- VGG: [Very Deep Convolutional Networks for Large-scale Image Recognition](https://arxiv.org/pdf/1409.1556), Karen Simonyan, Andrew Zisserman
- GoogLeNet: [Going Deeper with Convolutions](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf), Christian Szegedy1, Wei Liu2, Yangqing Jia
- Xception: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357), Franc ̧ois Chollet
- InceptionV4: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261), Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
- DarkNet: [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), Joseph Redmon, Ali Farhadi
- DenseNet: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993), Gao Huang, Zhuang Liu, Laurens van der Maaten
- SqueezeNet: [SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE](https://arxiv.org/abs/1602.07360), Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
- ResNeXt101_wsl: [Exploring the Limits of Weakly Supervised Pretraining](https://arxiv.org/abs/1805.00932), Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, Laurens van der Maaten
- Fix_ResNeXt101_wsl: [Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423), Hugo Touvron, Andrea Vedaldi, Matthijs Douze, Herve ́ Je ́gou

## Update

- 2018/12/03 **Stage1**: Update AlexNet, ResNet50, ResNet101, MobileNetV1
- 2018/12/23 **Stage2**: Update VGG Series, SeResNeXt50_32x4d, SeResNeXt101_32x4d, ResNet152
- 2019/01/31 Update MobileNetV2_x1_0
- 2019/04/01 **Stage3**: Update ResNet18, ResNet34, GoogLeNet, ShuffleNetV2
- 2019/06/12 **Stage4**:Update ResNet50_vc, ResNet50_vd, ResNet101_vd, ResNet152_vd, ResNet200_vd, SE154_vd InceptionV4, ResNeXt101_64x4d, ResNeXt101_vd_64x4d
- 2019/06/22 Update ResNet50_vd_v2
- 2019/07/02 **Stage5**: Update MobileNetV2_x0_5, ResNeXt50_32x4d, ResNeXt50_64x4d, Xception_41, ResNet101_vd
- 2019/07/19 **Stage6**: Update ShuffleNetV2_x0_25, ShuffleNetV2_x0_33, ShuffleNetV2_x0_5, ShuffleNetV2_x1_0, ShuffleNetV2_x1_5, ShuffleNetV2_x2_0, MobileNetV2_x0_25, MobileNetV2_x1_5, MobileNetV2_x2_0, ResNeXt50_vd_64x4d, ResNeXt101_32x4d, ResNeXt152_32x4d
- 2019/08/01 **Stage7**: Update DarkNet53, DenseNet121. Densenet161, DenseNet169, DenseNet201, DenseNet264, SqueezeNet1_0, SqueezeNet1_1, ResNeXt50_vd_32x4d, ResNeXt152_64x4d, ResNeXt101_32x8d_wsl, ResNeXt101_32x16d_wsl, ResNeXt101_32x32d_wsl, ResNeXt101_32x48d_wsl, Fix_ResNeXt101_32x48d_wsl

## Contribute

If you can fix an issue or add a new feature, please open a PR to us. If your PR is accepted, you can get scores according to the quality and difficulty of your PR(0~5), while you got 10 scores, you can contact us for interview or recommendation letter.
