# 图像分类以及模型库

---
## 内容
- [简介](#简介)
- [快速开始](#快速开始)
    - [安装说明](#安装说明)
    - [数据准备](#数据准备)
    - [模型训练](#模型训练)
    - [参数微调](#参数微调)
    - [模型评估](#模型评估)
    - [模型预测](#模型预测)
- [进阶使用](#进阶使用)
    - [混合精度训练](#混合精度训练)
    - [CE测试](#ce测试)
- [已发布模型及其性能](#已发布模型及其性能)
- [FAQ](#faq)
- [参考文献](#参考文献)
- [版本更新](#版本更新)
- [如何贡献代码](#如何贡献代码)
- [反馈](#反馈)

## 简介
图像分类是计算机视觉的重要领域，它的目标是将图像分类到预定义的标签。近期，许多研究者提出很多不同种类的神经网络，并且极大的提升了分类算法的性能。本页将介绍如何使用PaddlePaddle进行图像分类。

## 快速开始

### 安装说明
在当前目录下运行样例代码需要python 2.7及以上版本，PadddlePaddle Fluid v1.5或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据 [installation document](http://paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html) 中的说明来更新PaddlePaddle。

### 数据准备

下面给出了ImageNet分类任务的样例，首先，通过如下的方式进行数据的准备：
```
cd data/ILSVRC2012/
sh download_imagenet2012.sh
```
在```download_imagenet2012.sh```脚本中，通过下面三步来准备数据：

**步骤一：** 首先在```image-net.org```网站上完成注册，用于获得一对```Username```和```AccessKey```。

**步骤二：** 从ImageNet官网下载ImageNet-2012的图像数据。训练以及验证数据集会分别被下载到"train" 和 "val" 目录中。请注意，ImaegNet数据的大小超过40GB，下载非常耗时；已经自行下载ImageNet的用户可以直接将数据组织放置到```data/ILSVRC2012```。

**步骤三：** 下载训练与验证集合对应的标签文件。下面两个文件分别包含了训练集合与验证集合中图像的标签：

* train_list.txt: ImageNet-2012训练集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
train/n02483708/n02483708_2436.jpeg 369
```
* val_list.txt: ImageNet-2012验证集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
val/ILSVRC2012_val_00000001.jpeg 65
```
注意：可能需要根据本地环境调整reader.py相关路径来正确读取数据。

### 模型训练

数据准备完毕后，可以通过如下的方式启动训练：
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
**参数说明：**

* **model**: 模型名称， 默认值: "SE_ResNeXt50_32x4d"
* **num_epochs**: 训练回合数，默认值: 120
* **batch_size**: 批大小，默认值: 256
* **use_gpu**: 是否在GPU上运行，默认值: True
* **total_images**: 图片数，ImageNet2012默认值: 1281167.
* **class_dim**: 类别数，默认值: 1000
* **image_shape**: 图片大小，默认值: "3,224,224"
* **model_save_dir**: 模型存储路径，默认值: "output/"
* **with_mem_opt**: 是否开启显存优化，默认值: False
* **with_inplace**: 是否开启inplace显存优化，默认值: True
* **lr_strategy**: 学习率变化策略，默认值: "piecewise_decay"
* **lr**: 初始学习率，默认值: 0.1
* **pretrained_model**: 预训练模型路径，默认值: None
* **checkpoint**: 用于继续训练的检查点（指定具体模型存储路径，如"output/SE_ResNeXt50_32x4d/100/"），默认值: None
* **fp16**: 是否开启混合精度训练，默认值: False
* **scale_loss**: 调整混合训练的loss scale值，默认值: 1.0
* **l2_decay**: l2_decay值，默认值: 1e-4
* **momentum_rate**: momentum_rate值，默认值: 0.9
* **use_label_smoothing**: 是否对数据进行label smoothing处理，默认值:False
* **label_smoothing_epsilon**: label_smoothing的epsilon值，默认值:0.2
* **lower_scale**: 数据随机裁剪处理时的lower scale值, upper scale值固定为1.0，默认值:0.08
* **lower_ratio**: 数据随机裁剪处理时的lower ratio值，默认值:3./4.
* **upper_ration**: 数据随机裁剪处理时的upper ratio值，默认值:4./3.
* **resize_short_size**: 指定数据处理时改变图像大小的短边值，默认值: 256
* **use_mixup**: 是否对数据进行mixup处理，默认值:False
* **mixup_alpha**: 指定mixup处理时的alpha值，默认值: 0.2
* **is_distill**: 是否进行蒸馏训练，默认值: False

**在```run.sh```中有用于训练的脚本.**

**数据读取器说明：** 数据读取器定义在PIL：```reader.py```和CV2:```reader_cv2.py```文件中，现在默认基于cv2的数据读取器, 在[训练阶段](#模型训练), 默认采用的增广方式是随机裁剪与水平翻转, 而在[模型评估](#模型评估)与[模型预测](#模型预测)阶段用的默认方式是中心裁剪。当前支持的数据增广方式有：

* 旋转
* 颜色抖动（cv2暂未实现）
* 随机裁剪
* 中心裁剪
* 长宽调整
* 水平翻转

### 参数微调

参数微调是指在特定任务上微调已训练模型的参数。可以下载[已有模型及其性能](#已有模型及其性能)并且设置```path_to_pretrain_model```为模型所在路径，微调一个模型可以采用如下的命令：
```
python train.py \
       --pretrained_model=${path_to_pretrain_model}
```
注意：根据具体模型和任务添加并调整其他参数

### 模型评估
模型评估是指对训练完毕的模型评估各类性能指标。可以下载[已有模型及其性能](#已有模型及其性能)并且设置```path_to_pretrain_model```为模型所在路径。运行如下的命令，可以获得模型top-1/top-5精度:
```
python eval.py \
       --pretrained_model=${path_to_pretrain_model}
```
注意：根据具体模型和任务添加并调整其他参数

### 模型预测
模型预测可以获取一个模型的预测分数或者图像的特征，可以下载[已有模型及其性能](#已有模型及其性能)并且设置```path_to_pretrain_model```为模型所在路径。运行如下的命令获得预测分数，：
```
python infer.py \
       --pretrained_model=${path_to_pretrain_model}
```
注意：根据具体模型和任务添加并调整其他参数


##进阶使用

### 混合精度训练

可以通过开启`--fp16=True`启动混合精度训练，这样训练过程会使用float16数据，并输出float32的模型参数（"master"参数）。您可能需要同时传入`--scale_loss`来解决fp16训练的精度问题，通常传入`--scale_loss=8.0`即可。

注意，目前混合精度训练不能和内存优化功能同时使用，所以需要传`--with_mem_opt=False`这个参数来禁用内存优化功能。

### CE测试

注意：CE相关代码仅用于内部测试，enable_ce默认设置False。


## 已发布模型及其性能
表格中列出了在models目录下目前支持的图像分类模型，并且给出了已完成训练的模型在ImageNet-2012验证集合上的top-1/top-5精度，以及Paddle Fluid和Paddle TensorRT基于动态链接库的预测时间（测
试GPU型号为Tesla P4）。由于Paddle TensorRT对ShuffleNetV2_swish使用的激活函数swish，MobileNetV2使用的激活函数relu6不支持，因此预测加速不明显。可以通过点击相应模型的名称下载对应的预训练模型。

- 注意
   - 1：ResNet50_vd_v2是ResNet50_vd蒸馏版本。
   - 2：InceptionV4和Xception采用的输入图像的分辨率为299x299，DarkNet53为256x256，Fix_ResNeXt101_32x48d_wsl为320x320，其余模型使用的分辨率均为224x224。在预测时，DarkNet53与Fix_ResNeXt101_32x48d_wsl系列网络resize_short_size与输入的图像分辨率的宽或高相同,InceptionV4和Xception网络resize_short_size为320，其余网络resize_short_size均为256。
   - 3：调用动态链接库预测时需要将训练模型转换为二进制模型

    ```python infer.py --save_inference=True```
   - 4: ResNeXt101_wsl系列的预训练模型转自pytorch模型，详情请移步[RESNEXT WSL](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/)。


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

**Q:** 加载预训练模型报错，Enforce failed. Expected x_dims[1] == labels_dims[1], but received x_dims[1]:1000 != labels_dims[1]:6.

**A:** 维度对不上，删掉预训练参数中的FC

## 参考文献
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

## 版本更新
- 2018/12/03 **Stage1**: 更新AlexNet，ResNet50，ResNet101，MobileNetV1
- 2018/12/23 **Stage2**: 更新VGG系列 SeResNeXt50_32x4d，SeResNeXt101_32x4d，ResNet152
- 2019/01/31 更新MobileNetV2_x1_0
- 2019/04/01 **Stage3**: 更新ResNet18，ResNet34，GoogLeNet，ShuffleNetV2
- 2019/06/12 **Stage4**: 更新ResNet50_vc，ResNet50_vd，ResNet101_vd，ResNet152_vd，ResNet200_vd，SE154_vd InceptionV4，ResNeXt101_64x4d，ResNeXt101_vd_64x4d
- 2019/06/22 更新ResNet50_vd_v2
- 2019/07/02 **Stage5**: 更新MobileNetV2_x0_5, ResNeXt50_32x4d, ResNeXt50_64x4d, Xception_41, ResNet101_vd
- 2019/07/19 **Stage6**: 更新ShuffleNetV2_x0_25, ShuffleNetV2_x0_33, ShuffleNetV2_x0_5, ShuffleNetV2_x1_0, ShuffleNetV2_x1_5, ShuffleNetV2_x2_0, MobileNetV2_x0_25, MobileNetV2_x1_5, MobileNetV2_x2_0, ResNeXt50_vd_64x4d, ResNeXt101_32x4d, ResNeXt152_32x4d
- 2019/08/01 **Stage7**: 更新DarkNet53, DenseNet121. Densenet161, DenseNet169, DenseNet201, DenseNet264, SqueezeNet1_0, SqueezeNet1_1, ResNeXt50_vd_32x4d, ResNeXt152_64x4d, ResNeXt101_32x8d_wsl, ResNeXt101_32x16d_wsl, ResNeXt101_32x32d_wsl, ResNeXt101_32x48d_wsl, Fix_ResNeXt101_32x48d_wsl

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
