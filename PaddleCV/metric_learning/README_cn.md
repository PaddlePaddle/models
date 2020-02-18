# 深度度量学习
度量学习是一种对样本对学习区分性特征的方法，目的是在特征空间中，让同一个类别的样本具有较小的特征距离，不同类的样本具有较大的特征距离。随着深度学习技术的发展，基于深度神经网络的度量学习方法已经在许多视觉任务上提升了很大的性能，例如：人脸识别、人脸校验、行人重识别和图像检索等等。在本章节，介绍在PaddlePaddle Fluid里实现的几种度量学习方法和使用方法，具体包括[数据准备](#数据准备)，[模型训练](#模型训练)，[模型微调](#模型微调)，[模型评估](#模型评估)，[模型预测](#模型预测)。

---
## 简介
- [安装](#安装)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型微调](#模型微调)
- [模型评估](#模型评估)
- [模型预测](#模型预测)
- [模型性能](#模型性能)

## 安装

运行本章节代码需要在PaddlePaddle Fluid v0.14.0 或更高的版本环境。如果你的设备上的PaddlePaddle版本低于v0.14.0，请按照此[安装文档](http://paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/index_cn.html)进行安装和跟新。

## 数据准备

Stanford Online Product(SOP) 数据集下载自eBay，包含120053张商品图片，有22634个类别。我们使用该数据集进行实验。训练时，使用59551张图片，11318个类别的数据；测试时，使用60502张图片，11316个类别。首先，SOP数据集可以使用以下脚本下载：
```
cd data/
sh download_sop.sh
```

SOP数据集的目录结构如下图:
```
data/
├──Stanford_Online_Products
│   ├── bicycle_final
│   ├── bicycle_final.txt
│   ├── cabinet_final
│   ├── cabinet_final.txt
|   ...
├──Stanford_Online_Products.zip
```

如果你是Windows用户，你可以通过下面的ftp链接下载数据集，然后解压到```data```目录下。
```
SOP: ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
```

## 模型训练 

为了训练度量学习模型，我们需要一个神经网络模型作为骨架模型（如[ResNet50](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar)）和度量学习代价函数来进行优化。我们首先使用 softmax 或者 arcmargin 来进行训练，然后使用其它的代价函数来进行微调，例如：triplet，quadruplet和eml。下面是一个使用arcmargin训练的例子：


```
python train_elem.py  \
        --model=ResNet50 \
        --train_batch_size=256 \
        --test_batch_size=50 \
        --lr=0.01 \
        --total_iter_num=30000 \
        --use_gpu=True \
        --pretrained_model=${path_to_pretrain_imagenet_model} \
        --model_save_dir=${output_model_path} \
        --loss_name=arcmargin \
        --arc_scale=80.0 \ 
        --arc_margin=0.15 \
        --arc_easy_margin=False
```
**参数介绍:**
* **model**: 使用的模型名字. 默认: "ResNet50".
* **train_batch_size**: 训练的 mini-batch大小. 默认: 256.
* **test_batch_size**: 测试的 mini-batch大小. 默认: 50.
* **lr**: 初始学习率. 默认: 0.01.
* **total_iter_num**: 总的训练迭代轮数. 默认: 30000.
* **use_gpu**: 是否使用GPU. 默认: True.
* **pretrained_model**: 预训练模型的路径. 默认: None.
* **model_save_dir**: 保存模型的路径. 默认: "output".
* **loss_name**: 优化的代价函数. 默认: "softmax".
* **arc_scale**: arcmargin的参数. 默认: 80.0.
* **arc_margin**: arcmargin的参数. 默认: 0.15.
* **arc_easy_margin**: arcmargin的参数. 默认: False.

## 模型微调

网络微调是在指定的任务上加载已有的模型来微调网络。在用softmax和arcmargin训完网络后，可以继续使用triplet，quadruplet或eml来微调网络。下面是一个使用eml来微调网络的例子：

```
python train_pair.py  \
        --model=ResNet50 \
        --train_batch_size=160 \
        --test_batch_size=50 \
        --lr=0.0001 \
        --total_iter_num=100000 \
        --use_gpu=True \
        --pretrained_model=${path_to_pretrain_arcmargin_model} \
        --model_save_dir=${output_model_path} \
        --loss_name=eml \
        --samples_each_class=2
```

## 模型评估
模型评估主要是评估模型的检索性能。这里需要设置```path_to_pretrain_model```。可以使用下面命令来计算Recall@Rank-1。
```
python eval.py \
       --model=ResNet50 \
       --batch_size=50 \
       --pretrained_model=${path_to_pretrain_model} \
```

## 模型预测
模型预测主要是基于训练好的网络来获取图像数据的特征，下面是模型预测的例子：
```
python infer.py \
       --model=ResNet50 \
       --batch_size=1 \         
       --pretrained_model=${path_to_pretrain_model}
```

## 模型性能

下面列举了几种度量学习的代价函数在SOP数据集上的检索效果，这里使用Recall@Rank-1来进行评估。

|预训练模型 | softmax | arcmargin
|- | - | -:
|未微调 | 77.42% | 78.11%
|使用triplet微调 | 78.37% | 79.21%
|使用quadruplet微调 | 78.10% | 79.59%
|使用eml微调 | 79.32% | 80.11%
|使用npairs微调 | - | 79.81%

## 引用

- ArcFace: Additive Angular Margin Loss for Deep Face Recognition [链接](https://arxiv.org/abs/1801.07698)
- Margin Sample Mining Loss: A Deep Learning Based Method for Person Re-identification [链接](https://arxiv.org/abs/1710.00478)
- Large Scale Strongly Supervised Ensemble Metric Learning, with Applications to Face Verification and Retrieval [链接](https://arxiv.org/abs/1212.6094)
- Improved Deep Metric Learning with Multi-class N-pair Loss Objective [链接](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)
