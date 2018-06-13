
# 图像分类以及模型库
图像分类是计算机视觉的重要领域，它的目标是将图像分类到预定义的标签。近期，需要研究者提出很多不同种类的神经网络，并且极大的提升了分类算法的性能。本页将介绍如何使用PaddlePaddle进行图像分类，包括[数据准备](#data-preparation)、 [训练](#training-a-model)、[参数微调](#finetuning)、[模型评估](#evaluation)以及[模型推断](#inference)。

---
## 内容
- [安装](#installation)
- [数据准备](#data-preparation)
- [模型训练](#training-a-model)
- [参数微调](#finetuning)
- [模型评估](#evaluation)
- [模型推断](#inference)
- [已有模型及其性能](#supported-models)

## 安装

在当前目录下运行样例代码需要PadddlePaddle的v0.10.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)中的说明来更新PaddlePaddle。

## 数据准备

下面给出了ImageNet分类任务的样例，首先，通过如下的方式进行数据的准备：
```
cd data/ILSVRC2012/
sh download_imagenet2012.sh
```
在```download_imagenet2012.sh```脚本中，通过下面两步来准备数据：

**步骤一：** 从ImageNet官网下载ImageNet-2012的图像数据。训练以及验证数据集会分别被下载到"train" 和 "val" 目录中。

**步骤二：** 下载训练与验证集合对应的标签文件。下面两个文件分别包含了训练集合与验证集合中图像的标签：

* *train_list.txt*: ImageNet-2012训练集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
train/n02483708/n02483708_2436.jpeg 369
train/n03998194/n03998194_7015.jpeg 741
train/n04523525/n04523525_38118.jpeg 884
train/n04596742/n04596742_3032.jpeg 909
train/n03208938/n03208938_7065.jpeg 535
...
```
* *val_list.txt*: ImageNet-2012验证集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
val/ILSVRC2012_val_00000001.jpeg 65
val/ILSVRC2012_val_00000002.jpeg 970
val/ILSVRC2012_val_00000003.jpeg 230
val/ILSVRC2012_val_00000004.jpeg 809
val/ILSVRC2012_val_00000005.jpeg 516
...
```

## 模型训练

数据准备完毕后，可以通过如下的方式启动训练：
```
python train.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --total_images=1281167 \
       --class_dim=1000
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --with_mem_opt=False \
       --lr_strategy=piecewise_decay \
       --lr=0.1
```
**参数说明：**
* **model**: name model to use. Default: "SE_ResNeXt50_32x4d".
* **num_epochs**: the number of epochs. Default: 120.
* **batch_size**: the size of each mini-batch. Default: 256.
* **use_gpu**: whether to use GPU or not. Default: True.
* **total_images**: total number of images in the training set. Default: 1281167.
* **class_dim**: the class number of the classification task. Default: 1000.
* **image_shape**: input size of the network. Default: "3,224,224".
* **model_save_dir**: the directory to save trained model. Default: "output".
* **with_mem_opt**: whether to use memory optimization or not. Default: False.
* **lr_strategy**: learning rate changing strategy. Default: "piecewise_decay".
* **lr**: initialized learning rate. Default: 0.1.
* **pretrained_model**: model path for pretraining. Default: None.
* **checkpoint**: the checkpoint path to resume. Default: None.

**数据读取器说明：**

数据读取器定义在```reader.py```中。在[训练阶段](#training-a-model), 默认采用的增广方式是随机裁剪与水平翻转, 而在[评估](#inference)与[推断](#inference)阶段用的默认方式是中心裁剪。当前支持的数据增广方式有：
* 旋转
* 颜色抖动
* 随机裁剪
* 中心裁剪
* 长宽调整
* 水平翻转

## 参数微调

参数微调是指在特定任务上微调已训练模型的参数。通过初始化```path_to_pretrain_model```，微调一个模型可以采用如下的命令：
```
python train.py
       --model=SE_ResNeXt50_32x4d \
       --pretrained_model=${path_to_pretrain_model} \
       --batch_size=32 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --with_mem_opt=True \
       --lr_strategy=piecewise_decay \
       --lr=0.1
```

## 模型评估
模型评估是指对训练完毕的模型评估各类性能指标。运行如下的命令，可以获得一个模型top-1/top-5精度:
```
python eval.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=True \
       --pretrained_model=${path_to_pretrain_model}
```

## 模型推断
模型推断可以获取一个模型的预测分数或者图像的特征：
```
python infer.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=True \
       --pretrained_model=${path_to_pretrain_model}
```

## 已有模型及其性能

表格中列出了在"models"目录下支持的神经网络种类，并且给出了已完成训练的模型在ImageNet-2012验证集合上的top-1/top-5精度。预训练模型可以通过点击相应模型的名称进行下载。

|model | top-1/top-5 accuracy
|- | -:
|[AlexNet](http://paddle-imagenet-models.bj.bcebos.com/alexnet_model.tar) | 57.21%/79.72%
|VGG11 | -
|VGG13 | -
|VGG16 | -
|VGG19 | -
|GoogleNet | -
|InceptionV4 | -
|MobileNet | -
|[ResNet50](http://paddle-imagenet-models.bj.bcebos.com/resnet_50_model.tar) | 76.63%/93.10%
|ResNet101 | -
|ResNet152 | -
|[SE_ResNeXt50_32x4d](http://paddle-imagenet-models.bj.bcebos.com/se_resnext_50_model.tar) | 78.33%/93.96%
|SE_ResNeXt101_32x4d | -
|SE_ResNeXt152_32x4d | -
|DPN68 | -
|DPN92 | -
|DPN98 | -
|DPN107 | -
|DPN131 | -
