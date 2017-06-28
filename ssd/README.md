# SSD目标检测
## 概述
SSD全称为Single Shot MultiBox Detector，是目标检测领域较新且效果较好的检测算法之一，具体参见论文\[[1](#引用)\]。SSD算法主要特点是检测速度快且检测精度高，当输入图像大小为300*300，显卡采用Nvidia Titan X时，检测速度可达到59FPS，并且在VOC2007 test数据集上mAP高达74.3%。PaddlePaddle已集成SSD算法，本示例旨在介绍如何使用PaddlePaddle的SSD模型进行目标检测。下文展开顺序为：首先简要介绍SSD原理，然后介绍示例包含文件及作用，接着介绍如何在PASCAL VOC数据集上训练、评估及检测，最后简要介绍如何在自有数据集上使用SSD。
## SSD原理
SSD使用一个卷积神经网络实现“端到端”的检测，所谓“端到端”指输入为原始图像，输出为检测结果，无需借助外部工具或流程进行特征提取、候选框生成等。论文中SSD的基础模型为VGG-16，其在VGG-16的某些层后面增加了一些额外的层进行候选框的提取，下图为模型的总体结构：

<p align="center">
<img src="images/ssd_network.png" width="600" hspace='10'/> <br/>
图1. SSD网络结构
</p>

如图所示，候选框的生成规则是预先设定的，比如Conv7输出的特征图每个像素点会对应6个候选框，这些候选框长宽比或面积有区分。在预测阶段模型会对这些提取出来的候选框做后处理，然后输出作为最终的检测结果。
## 示例总览
本示例共包含如下文件：
<center>

文件 |  用途
---- | -----
train.py | 训练脚本
eval.py | 评估脚本，用于评估训好模型
infer.py | 检测脚本，给定图片及模型，实施检测
visual.py | 检测结果可视化
image_util.py | 图像预处理所需公共函数
data_provider.py | 数据处理脚本，生成训练、评估或检测所需数据
config/pascal\_voc\_conf.py | 神经网络超参数配置文件
data/label\_list | 类别列表
data/prepare\_voc\_data.py | 准备训练PASCAL VOC数据列表

</center>
<center>表1. 示例文件</center>

训练阶段需要对数据做预处理，包括裁剪、采样等，这部分操作在```image_util.py```和```data_provider.py```中完成；值得注意的是，```config/vgg_config.py```为参数配置文件，包括训练参数、神经网络参数等，本配置文件包含参数是针对PASCAL VOC数据配置的，当训练自有数据时，需要仿照该文件配置新的参数；```data/prepare_voc_data.py```脚本用来生成文件列表，包括切分训练集和测试集，使用时需要用户事先下载并解压数据，默认采用VOC2007和VOC2012。

## PASCAL VOC数据集
### 数据准备
首先需要下载数据集，VOC2007\[[2](#引用)\]和VOC2012\[[3](#引用)\]，VOC2007包含训练集和测试集，VOC2012只包含训练集，将下载好的数据解压，目录结构为```VOCdevkit/{VOC2007，VOC2012}```。进入```data```目录，运行```python prepare_voc_data.py```即可生成```trainval.txt```和```test.txt```，默认```prepare_voc_data.py```和```VOCdevkit```在相同目录下，且生成的文件列表也在该目录。需注意```trainval.txt```既包含VOC2007的训练数据，也包含VOC2012的训练数据，```test.txt```只包含VOC2007的测试数据。
### 预训练模型准备
下载训练好的VGG-16模型，推荐在ImageNet分类数据集上预训练的模型，针对caffe训练的模型，PaddlePaddle提供转换脚本，可方便转换成PaddlePaddle格式（待扩展），这里默认转换后的模型路径为```atrous_vgg/model.tar.gz```。
### 模型训练
直接执行```python train.py```即可进行训练。需要注意本示例仅支持CUDA GPU环境，无法在CPU上训练。```train.py```的一些关键执行逻辑：

```python
paddle.init(use_gpu=True, trainer_count=4)
data_args = data_provider.Settings(
                data_dir='./data',
                label_file='label_list',
                resize_h=cfg.IMG_HEIGHT,
                resize_w=cfg.IMG_WIDTH,
                mean_value=[104,117,124])
train(train_file_list='./data/trainval.txt',
      dev_file_list='./data/test.txt',
      data_args=data_args,
      init_model_path='./atrous_vgg/model.tar.gz')
```

调用```paddle.init```指定使用4卡GPU训练；调用```data_provider.Settings```配置数据预处理所需参数，其中```cfg.IMG_HEIGHT```和```cfg.IMG_WIDTH```在配置文件```config/vgg_config.py```中设置，这里均为300；调用```train```执行训练，其中```train_file_list```指定训练数据列表，```dev_file_list```指定评估数据列表，```init_model_path```指定预选模型位置。训练过程中会打印一些日志信息，每训练10个batch会输出当前的轮数、当前batch的cost及mAP，每训练一个pass，会保存一次模型，默认保存在```models```目录下（注：需事先创建）。

### 模型评估
### 图像检测

## 自有数据集

## 引用
1. Liu, Wei, et al. "Ssd: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016.
2. http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html
3. http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
