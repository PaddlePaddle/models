中文 | [English](README_en.md)

# 图像分类以及模型库

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
    - [Mixup训练](#mixup训练)
    - [混合精度训练](#混合精度训练)
    - [自定义数据集](#自定义数据集)
- [已发布模型及其性能](#已发布模型及其性能)
- [FAQ](#faq)
- [参考文献](#参考文献)
- [版本更新](#版本更新)
- [如何贡献代码](#如何贡献代码)

---

## 简介
图像分类是计算机视觉的重要领域，它的目标是将图像分类到预定义的标签。近期，许多研究者提出很多不同种类的神经网络，并且极大的提升了分类算法的性能。本页将介绍如何使用PaddlePaddle进行图像分类。

同时推荐用户参考[ IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/122278)

## 快速开始

### 安装说明

在当前目录下运行样例代码需要python 2.7及以上版本，PadddlePaddle Fluid v1.6或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据 [安装文档](http://paddlepaddle.org/documentation/docs/zh/1.6/beginners_guide/install/index_cn.html) 中的说明来更新PaddlePaddle。

#### 环境依赖

python >= 2.7，CUDA >= 8.0，CUDNN >= 7.0
运行训练代码需要安装numpy，cv2

```bash
pip install opencv-python
pip install numpy
```

### 数据准备

下面给出了ImageNet分类任务的样例，首先，通过如下的方式进行数据的准备：
```
cd data/ILSVRC2012/
sh download_imagenet2012.sh
```
在```download_imagenet2012.sh```脚本中，通过下面三步来准备数据：

**步骤一：** 首先在```image-net.org```网站上完成注册，用于获得一对```Username```和```AccessKey```。

**步骤二：** 从ImageNet官网下载ImageNet-2012的图像数据。训练以及验证数据集会分别被下载到"train" 和 "val" 目录中。注意，ImageNet数据的大小超过140GB，下载非常耗时；已经自行下载ImageNet的用户可以直接将数据组织放置到```data/ILSVRC2012```。

**步骤三：** 下载训练与验证集合对应的标签文件。下面两个文件分别包含了训练集合与验证集合中图像的标签：

* train_list.txt: ImageNet-2012训练集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
train/n02483708/n02483708_2436.jpeg 369
```
* val_list.txt: ImageNet-2012验证集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
val/ILSVRC2012_val_00000001.jpeg 65
```
注意：可能需要根据本地环境调整reader.py中相关路径来正确读取数据。

### 模型训练

数据准备完毕后，可以通过如下的方式启动训练：
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

注意: 当添加如step_epochs这种列表型参数，需要去掉"="，如：--step_epochs 10 20 30

或通过run.sh 启动训练

```bash
bash run.sh train 模型名
```

**多进程模型训练：**

如果你有多张GPU卡的话，我们强烈建议你使用多进程模式来训练模型，这会极大的提升训练速度。启动方式如下：
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
或者参考 scripts/train/ResNet50_dist.sh

**参数说明：**

环境配置部分：

* **data_dir**: 数据存储路径，默认值: "./data/ILSVRC2012/"
* **model_save_dir**: 模型存储路径，默认值: "output/"
* **pretrained_model**: 加载预训练模型路径，默认值: None
* **checkpoint**: 加载用于继续训练的检查点（指定具体模型存储路径，如"output/ResNet50/100/"），默认值: None
* **print_step**: 打印训练信息的batch步数，默认值：10
* **save_step**: 保存模型的epoch步数，默认值：1

模型类型和超参配置：

* **model**: 模型名称， 默认值: "ResNet50"
* **total_images**: 图片数，ImageNet2012，默认值: 1281167
* **class_dim**: 类别数，默认值: 1000
* **image_shape**: 图片大小，默认值: "3,224,224"
* **num_epochs**: 训练回合数，默认值: 120
* **batch_size**: batch size大小(所有设备)，默认值: 8
* **test_batch_size**: 测试batch大小，默认值：16
* **lr_strategy**: 学习率变化策略，默认值: "piecewise_decay"
* **lr**: 初始学习率，默认值: 0.1
* **l2_decay**: l2_decay值，默认值: 1e-4
* **momentum_rate**: momentum_rate值，默认值: 0.9
* **step_epochs**: piecewise dacay的decay step，默认值：[30,60,90]
* **decay_epochs**: exponential decay的间隔epoch数, 默认值: 2.4.
* **decay_rate**: exponential decay的下降率, 默认值: 0.97.

数据读取器和预处理配置：

* **lower_scale**: 数据随机裁剪处理时的lower scale值， upper scale值固定为1.0，默认值:0.08
* **lower_ratio**: 数据随机裁剪处理时的lower ratio值，默认值:3./4.
* **upper_ratio**: 数据随机裁剪处理时的upper ratio值，默认值:4./3.
* **resize_short_size**: 指定数据处理时改变图像大小的短边值，默认值: 256
* **crop_size**: 指定裁剪的大小，默认值:224
* **use_mixup**: 是否对数据进行mixup处理，默认值: False
* **mixup_alpha**: 指定mixup处理时的alpha值，默认值: 0.2
* **use_aa**: 是否对数据进行auto augment处理. 默认值: False.
* **reader_thread**: 多线程reader的线程数量，默认值: 8
* **reader_buf_size**: 多线程reader的buf_size， 默认值: 2048
* **interpolation**: 插值方法， 默认值：None
* **image_mean**: 图片均值，默认值：[0.485, 0.456, 0.406]
* **image_std**: 图片std，默认值：[0.229, 0.224, 0.225]


一些开关：

* **use_gpu**: 是否在GPU上运行，默认值: True
* **use_label_smoothing**: 是否对数据进行label smoothing处理，默认值: False
* **label_smoothing_epsilon**: label_smoothing的epsilon， 默认值:0.1
* **random_seed**: 随机数种子， 默认值: 1000
* **padding_type**: efficientNet中卷积操作的padding方式, 默认值: "SAME".
* **use_ema**: 是否在更新模型参数时使用ExponentialMovingAverage. 默认值: False.
* **ema_decay**: ExponentialMovingAverage的decay rate. 默认值: 0.9999.

**数据读取器说明：** 数据读取器定义在```reader.py```文件中，现在默认基于cv2的数据读取器， 在[训练阶段](#模型训练)，默认采用的增广方式是随机裁剪与水平翻转， 而在[模型评估](#模型评估)与[模型预测](#模型预测)阶段用的默认方式是中心裁剪。当前支持的数据增广方式有：

* 旋转
* 颜色抖动（暂未实现）
* 随机裁剪
* 中心裁剪
* 长宽调整
* 水平翻转
* 自动增广

### 参数微调

参数微调(Finetune)是指在特定任务上微调已训练模型的参数。可以下载[已发布模型及其性能](#已发布模型及其性能)并且设置```path_to_pretrain_model```为模型所在路径，微调一个模型可以采用如下的命令：

```bash
python train.py \
       --model=model_name \
       --pretrained_model=${path_to_pretrain_model}
```
注意：根据具体模型和任务添加并调整其他参数

### 模型评估

模型评估(Eval)是指对训练完毕的模型评估各类性能指标。可以下载[已发布模型及其性能](#已发布模型及其性能)并且设置```path_to_pretrain_model```为模型所在路径。运行如下的命令，可以获得模型top-1/top-5精度:

```bash
python eval.py \
       --model=model_name \
       --pretrained_model=${path_to_pretrain_model}
```
注意：根据具体模型和任务添加并调整其他参数

### 指数滑动平均的模型评估

注意: 如果你使用指数滑动平均来训练模型(--use_ema=True)，并且想要评估指数滑动平均后的模型，需要使用ema_clean.py将训练中保存下来的ema模型名字转换成原始模型参数的名字。

```
python ema_clean.py \
       --ema_model_dir=your_ema_model_dir \
       --cleaned_model_dir=your_cleaned_model_dir

python eval.py \
       --model=model_name \
       --pretrained_model=your_cleaned_model_dir
```

### 模型预测

模型预测(Infer)可以获取一个模型的预测分数或者图像的特征，可以下载[已发布模型及其性能](#已发布模型及其性能)并且设置```path_to_pretrain_model```为模型所在路径。运行如下的命令获得预测结果：

**参数说明：**

* **save_inference**: 是否保存模型，默认值：False
* **topk**: 按照置信由高到低排序标签结果，返回的结果数量，默认值：1
* **label_path**: 可读标签文件路径，默认值："./utils/tools/readable_label.txt"

```bash
python infer.py \
       --model=model_name \
       --pretrained_model=${path_to_pretrain_model}
```
注意：根据具体模型和任务添加并调整其他参数

模型预测默认ImageNet1000类类别，标签文件存储在/utils/tools/readable_label.txt中，如果使用自定义数据，请指定--label_path参数


## 进阶使用

### Mixup训练

训练中指定 --use_mixup=True 开启Mixup训练，本模型库中所有后缀为_vd的模型即代表开启Mixup训练

Mixup相关介绍参考[mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

### 混合精度训练

FP16相关内容已经迁移至PaddlePaddle/Fleet 中

### 自定义数据集

PaddlePaddle/Models ImageClassification 支持自定义数据

1. 组织自定义数据，调整数据读取器以正确的传入数据
2. 注意更改训练脚本中
--data_dim 类别数为自定义数据类别数
--total_image 图片数量
3. 当进行finetune时，
指定--pretrained_model 加载预训练模型，注意：本模型库提供的是基于ImageNet 1000类数据的预训练模型，当使用不同类别数的数据时，请删除预训练模型中fc_weight 和fc_offset参数


## 已发布模型及其性能
表格中列出了在models目录下目前支持的图像分类模型，并且给出了已完成训练的模型在ImageNet-2012验证集合上的top-1和top-5精度，以及Paddle Fluid和Paddle TensorRT基于动态链接库的预测时间（测试GPU型号为NVIDIA® Tesla® P4）。
可以通过点击相应模型的名称下载对应的预训练模型。

- 注意
   - 1：ResNet50_vd_v2是ResNet50_vd蒸馏版本。
   - 2：除EfficientNet外，InceptionV4和Xception采用的输入图像的分辨率为299x299，DarkNet53为256x256，Fix_ResNeXt101_32x48d_wsl为320x320，其余模型使用的分辨率均为224x224。在预测时，DarkNet53与Fix_ResNeXt101_32x48d_wsl系列网络resize_short_size与输入的图像分辨率的宽或高相同，InceptionV4和Xception网络resize_short_size为320，其余网络resize_short_size均为256。
   - 3: EfficientNetB0~B7的分辨率大小分别为224x224，240x240，260x260，300x300，380x380，456x456，528x528，600x600，预测时的resize_short_size在其分辨率的长或高的基础上加32，如EfficientNetB1的resize_short_size为272，在该系列模型训练和预测的过程中，图片resize参数interpolation的值设置为2（cubic插值方式），该模型在训练过程中使用了指数滑动平均策略，具体请参考[指数滑动平均](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/optimizer_cn.html#exponentialmovingaverage)。
   - 4：调用动态链接库预测时需要将训练模型转换为二进制模型。

        ```bash
        python infer.py \
               --model=model_name \
               --pretrained_model=${path_to_pretrain_model} \
               --save_inference=True
        ```

   - 5: ResNeXt101_wsl系列的预训练模型转自pytorch模型，详情见[ResNeXt wsl](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/)。


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
|[ResNet50_vd_v2](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_v2_pretrained.tar) | 79.84% | 94.93% | 9.058 | 5.259 |
|[ResNet101](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) | 77.56% | 93.64% | 15.447 | 8.473 |
|[ResNet101_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar) | 80.17% | 94.97% | 15.685 | 8.574 |
|[ResNet152](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_pretrained.tar) | 78.26% | 93.96% | 21.816 | 11.646 |
|[ResNet152_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_vd_pretrained.tar) | 80.59% | 95.30% | 22.041 | 11.858 |
|[ResNet200_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet200_vd_pretrained.tar) | 80.93% | 95.33% | 28.015 | 14.896 |

### ResNeXt Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_32x4d_pretrained.tar) | 77.75% | 93.82% | 12.863 | 9.241 |
|[ResNeXt50_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_vd_32x4d_pretrained.tar) | 79.56% | 94.62% | 13.673 | 9.162 |
|[ResNeXt50_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_64x4d_pretrained.tar) | 78.43% | 94.13% | 28.162 | 15.935 |
|[ResNeXt50_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_vd_64x4d_pretrained.tar) | 80.12% | 94.86% | 20.888 | 15.938 |
|[ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x4d_pretrained.tar) | 78.65% | 94.19% | 24.154 | 17.661 |
|[ResNeXt101_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_vd_32x4d_pretrained.tar) | 80.33% | 95.12% | 24.701 | 17.249 |
|[ResNeXt101_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_64x4d_pretrained.tar) | 78.43% | 94.13% | 41.073 | 31.288 |
|[ResNeXt101_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_vd_64x4d_pretrained.tar) | 80.78% | 95.20% | 42.277 | 32.620 |
|[ResNeXt152_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_32x4d_pretrained.tar) | 78.98% | 94.33% | 37.007 | 26.981 |
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
|[EfficientNetB1](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB1_pretrained.tar)<sup>[1](#trans)</sup> | 79.15% | 94.41% | 15.626 | 6.502 |
|[EfficientNetB2](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB2_pretrained.tar)<sup>[1](#trans)</sup> | 79.85% | 94.74% | 17.847 | 7.558 |
|[EfficientNetB3](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB3_pretrained.tar)<sup>[1](#trans)</sup> | 81.15% | 95.41% | 25.993 | 10.937 |
|[EfficientNetB4](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB4_pretrained.tar)<sup>[1](#trans)</sup> | 82.85% | 96.23% | 47.734 | 18.536 |
|[EfficientNetB5](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB5_pretrained.tar)<sup>[1](#trans)</sup> | 83.62% | 96.72% | 88.578 | 32.102 |
|[EfficientNetB6](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB6_pretrained.tar)<sup>[1](#trans)</sup> | 84.00% | 96.88% | 138.670 | 51.059 |
|[EfficientNetB7](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB7_pretrained.tar)<sup>[1](#trans)</sup> | 84.30% | 96.89% | 234.364 | 82.107 |

<a name="trans">[1]</a> 表示该预训练权重是由[官方的代码仓库](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)转换来的。

## FAQ

**Q:** 加载预训练模型报错，Enforce failed. Expected x_dims[1] == labels_dims[1], but received x_dims[1]:1000 != labels_dims[1]:6.

**A:** 类别数匹配不上，删掉最后一层分类层FC

**Q:** reader中报错AttributeError: 'NoneType' object has no attribute 'shape'

**A:** 文件路径load错误

**Q:** 出现cudaStreamSynchronize an illegal memory access was encountered errno:77 错误

**A:** 可能是因为显存问题导致，添加如下环境变量：

    export FLAGS_fast_eager_deletion_mode=1
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98

## 参考文献
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

## 版本更新
- 2018/12/03 **Stage1**: 更新AlexNet，ResNet50，ResNet101，MobileNetV1
- 2018/12/23 **Stage2**: 更新VGG系列，SeResNeXt50_32x4d，SeResNeXt101_32x4d，ResNet152
- 2019/01/31 更新MobileNetV2_x1_0
- 2019/04/01 **Stage3**: 更新ResNet18，ResNet34，GoogLeNet，ShuffleNetV2
- 2019/06/12 **Stage4**: 更新ResNet50_vc，ResNet50_vd，ResNet101_vd，ResNet152_vd，ResNet200_vd，SE154_vd InceptionV4，ResNeXt101_64x4d，ResNeXt101_vd_64x4d
- 2019/06/22 更新ResNet50_vd_v2
- 2019/07/02 **Stage5**: 更新MobileNetV2_x0_5，ResNeXt50_32x4d，ResNeXt50_64x4d，Xception41，ResNet101_vd
- 2019/07/19 **Stage6**: 更新ShuffleNetV2_x0_25，ShuffleNetV2_x0_33，ShuffleNetV2_x0_5，ShuffleNetV2_x1_0，ShuffleNetV2_x1_5，ShuffleNetV2_x2_0，MobileNetV2_x0_25，MobileNetV2_x1_5，MobileNetV2_x2_0，ResNeXt50_vd_64x4d，ResNeXt101_32x4d，ResNeXt152_32x4d
- 2019/08/01 **Stage7**: 更新DarkNet53，DenseNet121，Densenet161，DenseNet169，DenseNet201，DenseNet264，SqueezeNet1_0，SqueezeNet1_1，ResNeXt50_vd_32x4d，ResNeXt152_64x4d，ResNeXt101_32x8d_wsl，ResNeXt101_32x16d_wsl，ResNeXt101_32x32d_wsl，ResNeXt101_32x48d_wsl，Fix_ResNeXt101_32x48d_wsl
- 2019/09/11 **Stage8**: 更新ResNet18_vd，ResNet34_vd，MobileNetV1_x0_25，MobileNetV1_x0_5，MobileNetV1_x0_75，MobileNetV2_x0_75，MobilenNetV3_small_x1_0，DPN68，DPN92，DPN98，DPN107，DPN131，ResNeXt101_vd_32x4d，ResNeXt152_vd_64x4d，Xception65，Xception71，Xception41_deeplab，Xception65_deeplab，SE_ResNet50_vd
- 2019/09/20 更新EfficientNet

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
