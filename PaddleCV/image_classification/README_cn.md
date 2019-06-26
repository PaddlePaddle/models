# 图像分类以及模型库
图像分类是计算机视觉的重要领域，它的目标是将图像分类到预定义的标签。近期，许多研究者提出很多不同种类的神经网络，并且极大的提升了分类算法的性能。本页将介绍如何使用PaddlePaddle进行图像分类。

---
## 内容
- [安装](#安装)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [混合精度训练](#混合精度训练)
- [参数微调](#参数微调)
- [模型评估](#模型评估)
- [模型预测](#模型预测)
- [已有模型及其性能](#已有模型及其性能)

## 安装

在当前目录下运行样例代码需要PadddlePaddle Fluid的v0.13.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据 [installation document](http://paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/index_cn.html) 中的说明来更新PaddlePaddle。

注意：由于windows不支持nccl，当使用Windows GPU环境时候，需要将示例代码中的[fluid.ParallelExecutor](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/fluid_cn.html#parallelexecutor)替换为[fluid.Executor](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/fluid_cn.html#executor)。

## 数据准备

下面给出了ImageNet分类任务的样例，首先，通过如下的方式进行数据的准备：
```
cd data/ILSVRC2012/
sh download_imagenet2012.sh
```
在```download_imagenet2012.sh```脚本中，通过下面三步来准备数据：

**步骤一：** 首先在```image-net.org```网站上完成注册，用于获得一对```Username```和```AccessKey```。

**步骤二：** 从ImageNet官网下载ImageNet-2012的图像数据。训练以及验证数据集会分别被下载到"train" 和 "val" 目录中。请注意，ImaegNet数据的大小超过40GB，下载非常耗时；已经自行下载ImageNet的用户可以直接将数据组织放置到```data/ILSVRC2012```。

**步骤三：** 下载训练与验证集合对应的标签文件。下面两个文件分别包含了训练集合与验证集合中图像的标签：

* *train_list.txt*: ImageNet-2012训练集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
train/n02483708/n02483708_2436.jpeg 369
train/n03998194/n03998194_7015.jpeg 741
train/n04523525/n04523525_38118.jpeg 884
...
```
* *val_list.txt*: ImageNet-2012验证集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
val/ILSVRC2012_val_00000001.jpeg 65
val/ILSVRC2012_val_00000002.jpeg 970
val/ILSVRC2012_val_00000003.jpeg 230
...
```
注意：需要根据本地环境调整reader.py相关路径来正确读取数据。

## 模型训练

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

在```run.sh```中有用于训练的脚本.

**数据读取器说明：** 数据读取器定义在```reader.py```和```reader_cv2.py```中。一般, CV2可以提高数据读取速度, PIL reader可以得到相对更高的精度, 我们现在默认基于cv2的数据读取器, 在[训练阶段](#模型训练), 默认采用的增广方式是随机裁剪与水平翻转, 而在[模型评估](#模型评估)与[模型预测](#模型预测)阶段用的默认方式是中心裁剪。当前支持的数据增广方式有：
* 旋转
* 颜色抖动
* 随机裁剪
* 中心裁剪
* 长宽调整
* 水平翻转

## 混合精度训练

可以通过开启`--fp16=True`启动混合精度训练，这样训练过程会使用float16数据，并输出float32的模型参数（"master"参数）。您可能需要同时传入`--scale_loss`来解决fp16训练的精度问题，通常传入`--scale_loss=8.0`即可。

注意，目前混合精度训练不能和内存优化功能同时使用，所以需要传`--with_mem_opt=False`这个参数来禁用内存优化功能。

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
模型评估是指对训练完毕的模型评估各类性能指标。用户可以下载[已有模型及其性能](#已有模型及其性能)并且设置```path_to_pretrain_model```为模型所在路径。运行如下的命令，可以获得一个模型top-1/top-5精度:
```
python eval.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=True \
       --pretrained_model=${path_to_pretrain_model}
```

## 模型预测
模型预测可以获取一个模型的预测分数或者图像的特征：
```
python infer.py \
       --model=SE_ResNeXt50_32x4d \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=True \
       --pretrained_model=${path_to_pretrain_model}
```

## 已有模型及其性能
表格中列出了在models目录下目前支持的图像分类模型，并且给出了已完成训练的模型在ImageNet-2012验证集合上的top-1/top-5精度，以及Paddle Fluid和Paddle TensorRT基于动态链接库的预测时间（测
试GPU型号为Tesla P4）。可以通过点击相应模型的名称下载对应的预训练模型。基于动态链接库的预测方法，请参考http://www.paddlepaddle.org/documentation/docs/zh/1.4/advanced_usage/deploy/index_cn.html  由于Paddle TensorRT对ShuffleNetV2使用的激活函数swish，MobileNetV2使用的激活函数relu6不支持，因此预测加速不明显，Paddle TensorRT不久后添加对这两个op的支持。
- 注意1：ResNet50_vd_v2是ResNet50_vd蒸馏版本。
- 注意2：除了InceptionV4采用的输入图像的分辨率为299x299，其余模型测试时使用的分辨率均为224x224。
- 注意3：调用动态链接库预测时需要将训练模型转换为二进制模型，转换方法如下：a.将infer.py中参数save_inference设置为True; b.执行infer.py。
- Released models: specify parameter names

|model | top-1/top-5 accuracy(CV2) | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |
|[AlexNet](http://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.tar) | 56.72%/79.17% | 3.083 | 2.728 |
|[VGG11](https://paddle-imagenet-models-name.bj.bcebos.com/VGG11_pretrained.tar) | 69.28%/89.09% | 8.223 | 6.821 |
|[VGG13](https://paddle-imagenet-models-name.bj.bcebos.com/VGG13_pretrained.tar) | 70.02%/89.42% | 9.512 | 7.783 |
|[VGG16](https://paddle-imagenet-models-name.bj.bcebos.com/VGG16_pretrained.tar) | 72.00%/90.69% | 11.315 | 9.067 |
|[VGG19](https://paddle-imagenet-models-name.bj.bcebos.com/VGG19_pretrained.tar) | 72.56%/90.93% | 13.096 | 10.388 |
|[MobileNetV1](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) | 70.99%/89.68% | 2.609 | 1.615 |
|[MobileNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) | 72.15%/90.65% | 4.546 | 5.278 |
|[ResNet18](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar) | 70.98%/89.92% | 3.456 | 2.484 |
|[ResNet34](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar) | 74.57%/92.14% | 5.668 | 3.767 |
|[ResNet50](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar) | 76.50%/93.00% | 8.787 | 5.434 |
|[ResNet50_vc](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vc_pretrained.tar) |78.35%/94.03% | 9.013 | 5.463 |
|[ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar) | 79.12%/94.44% | 9.058 | 5.510 |
|[ResNet50_vd_v2](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_v2_pretrained.tar) | 79.84%/94.93% | 9.058 | 5.510 |
|[ResNet101](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) | 77.56%/93.64% | 15.447 | 8.779 |
|[ResNet101_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar) | 79.44%/94.47% | 15.685 | 8.878 |
|[ResNet152](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_pretrained.tar) | 78.26%/93.96% | 21.816 | 12.148 |
|[ResNet152_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_vd_pretrained.tar) | 80.59%/95.30% | 22.041 | 12.259 |
|[ResNet200_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet200_vd_pretrained.tar) | 80.93%/95.33% | 28.015 | 15.278 |
|[ResNeXt101_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_64x4d_pretrained.tar) | 79.35%/94.52% | 41.073 |  38.736 |
|[ResNeXt101_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_vd_64x4d_pretrained.tar) | 80.78%/95.20% | 42.277 | 40.929 |
|[SE_ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt50_32x4d_pretrained.tar) | 78.44%/93.96% | 14.916 | 12.126 |
|[SE_ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt101_32x4d_pretrained.tar) | 79.12%/94.20% | 30.085 | 24.110 |
|[SE154_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SE154_vd_pretrained.tar) | 81.40%/95.48% | 71.892 | 64.855 |
|[GoogleNet](https://paddle-imagenet-models-name.bj.bcebos.com/GoogleNet_pretrained.tar) | 70.70%/89.66% | 6.528 | 3.076 |
|[ShuffleNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar) | 70.03%/89.17% | 6.078 | 6.282 |
|[InceptionV4](https://paddle-imagenet-models-name.bj.bcebos.com/InceptionV4_pretrained.tar) | 80.77%/95.26% | 32.413 | 18.154 |
