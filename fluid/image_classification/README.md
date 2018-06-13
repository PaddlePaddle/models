# Image Classification and Model Zoo
Image classification, which is an important field of computer vision, is to classify an image into pre-defined labels. Recently, many researchers developed different kinds of neural networks and highly improve the classification performance. This page introduces how to do image classification with PaddlePaddle, including [data preparation](#data-preparation), [training](#training-a-model), [finetuning](#finetuning), [evaluation](#evaluation) and [inference](#inference).

---
## Table of Contents
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training a model with flexible parameters](#training-a-model)
- [Finetuning](#finetuning)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Supported models and performances](#supported-models)

## Installation

Running sample code in this directory requires PaddelPaddle v0.10.0 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in [installation document](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html) and make an update.

## Data preparation

An example for ImageNet classification is as follows. First of all, preparation of imagenet data can be done with two steps:

**step-1:** Download ImageNet-2012 dataset from website
```
cd data/
mkdir -p ILSVRC2012/
cd ILSVRC2012/
wget paddl_imagenet2012_dataset_url/ImageNet2012_dataset.tar
tar xf ImageNet2012_dataset.tar
```

**step-2:** Download training and validation label files
```
wget paddl_imagenet2012_label_url/ImageNet2012_label.tar
tar xf ImageNet2012_label.tar
```
there are two label files which contain train and validation image labels respectively:

* *train_list.txt*: label file imagenet-2012 training set, with each line seperated by SPACE, like:
```
train/n02483708/n02483708_2436.jpeg 369
train/n03998194/n03998194_7015.jpeg 741
train/n04523525/n04523525_38118.jpeg 884
train/n04596742/n04596742_3032.jpeg 909
train/n03208938/n03208938_7065.jpeg 535
...
```
* *val_list.txt*: label file of imagenet-2012 validation set, with each line seperated by SPACE, like.
```
val/ILSVRC2012_val_00000001.jpeg 65
val/ILSVRC2012_val_00000002.jpeg 970
val/ILSVRC2012_val_00000003.jpeg 230
val/ILSVRC2012_val_00000004.jpeg 809
val/ILSVRC2012_val_00000005.jpeg 516
...
```

## Training a model with flexible parameters

After data preparation, one can start  the training by:

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
**parameter introduction:**
* **model**: name model to use. Default: "SE_ResNeXt50_32x4d".
* **num_epochs**: the number of epochs. Default: 120.
* **batch_size**: the size of each mini-batch. Default: 256.
* **total_images**: total number of images in the training set. Default: 1281167.
* **class_dim**: the class number of the classification task. Default: 1000.
* **image_shape**: input size of the network. Default: "3,224,224".
* **model_save_dir**: the directory to save trained model. Default: "output".
* **with_mem_opt**: whether to use memory optimization or not. Default: False.
* **lr_strategy**: learning rate changing strategy. Default: "piecewise_decay".
* **lr**: initialized learning rate. Default: 0.1.
* **pretrained_model**: model path for pretraining. Default: None.
* **checkpoint**: the checkpoint path to resume. Default: None.

**data reader introduction:**

Data reader is defined in ```reader.py```. In [training stage](#training-a-model), random crop and flipping are used, while center crop is used in [evaluation](#inference) and [inference](#inference) stages. Supported data augmentation includes:
* rotation
* color jitter
* random crop
* center crop
* resize
* flipping

## Finetuning
```
python train.py
       --model=SE_ResNeXt50_32x4d \
       --pretrained_model=${path_to_pretrain_model} \
       --batch_size=32 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --with_mem_opt=False \
       --lr_strategy=piecewise_decay \
       --lr=0.1
```

## Evaluation
```
python eval.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=False \
       --pretrained_model=${path_to_pretrain_model}
```

## Inference
```
python infer.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=False \
       --pretrained_model=${path_to_pretrain_model}
```

## Supported models and performances

Models are trained by starting with learning rate ```0.1``` and decaying it by ```0.1``` after each ```30``` epoches, if not special introduced. Available top-1/top-5 validation accuracy on ImageNet 2012 is listed in table. Pretrained models can be downloaded by clicking related model names.

|model | top-1/top-5 accuracy
|- | -:
|[AlexNet](http://paddle-imagenet-models.bj.bcebos.com/alexnet_model.tar) | 56.65%/79.20%
|VGG11 | -
|VGG13 | -
|VGG16 | -
|VGG19 | -
|GoogleNet | -
|InceptionV4 | -
|MobileNet | -
|[ResNet50](http://paddle-imagenet-models.bj.bcebos.com/resnet_50_model.tar) | 76.44%/93.21%
|ResNet101 | -
|ResNet152 | -
|[SE_ResNeXt50_32x4d](http://paddle-imagenet-models.bj.bcebos.com/se_resnext_50_model.tar) | 78.25%/93.97%
|SE_ResNeXt101_32x4d | -
|SE_ResNeXt152_32x4d | -
|DPN68 | -
|DPN92 | -
|DPN98 | -
|DPN107 | -
|DPN131 | -
