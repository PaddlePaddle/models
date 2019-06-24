# Image Classification and Model Zoo
Image classification, which is an important field of computer vision, is to classify an image into pre-defined labels. Recently, many researchers developed different kinds of neural networks and highly improve the classification performance. This page introduces how to do image classification with PaddlePaddle Fluid.

---
## Table of Contents
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training a model with flexible parameters](#training-a-model-with-flexible-parameters)
- [Using Mixed-Precision Training](#using-mixed-precision-training)
- [Finetuning](#finetuning)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Supported models and performances](#supported-models-and-performances)

## Installation

Running sample code in this directory requires PaddelPaddle Fluid v0.13.0 and later, the latest release version is recommended, If the PaddlePaddle on your device is lower than v0.13.0, please follow the instructions in [installation document](http://paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/index_cn.html) and make an update.

Note: Please replace [fluid.ParallelExecutor](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/fluid_cn.html#parallelexecutor) to [fluid.Executor](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/fluid_cn.html#executor) when running the program in the windows & GPU environment.

## Data preparation

An example for ImageNet classification is as follows. First of all, preparation of imagenet data can be done as:
```
cd data/ILSVRC2012/
sh download_imagenet2012.sh
```

In the shell script ```download_imagenet2012.sh```,  there are three steps to prepare data:

**step-1:** Register at ```image-net.org``` first in order to get a pair of ```Username``` and ```AccessKey```, which are used to download ImageNet data.

**step-2:** Download ImageNet-2012 dataset from website. The training and validation data will be downloaded into folder "train" and "val" respectively. Please note that the size of data is more than 40 GB, it will take much time to download. Users who have downloaded the ImageNet data can organize it into ```data/ILSVRC2012``` directly.

**step-3:** Download training and validation label files. There are two label files which contain train and validation image labels respectively:

* *train_list.txt*: label file of imagenet-2012 training set, with each line seperated by ```SPACE```, like:
```
train/n02483708/n02483708_2436.jpeg 369
train/n03998194/n03998194_7015.jpeg 741
train/n04523525/n04523525_38118.jpeg 884
...
```
* *val_list.txt*: label file of imagenet-2012 validation set, with each line seperated by ```SPACE```, like.
```
val/ILSVRC2012_val_00000001.jpeg 65
val/ILSVRC2012_val_00000002.jpeg 970
val/ILSVRC2012_val_00000003.jpeg 230
...
```

You may need to modify the path in reader.py to load data correctly.

## Training a model with flexible parameters

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

**data reader introduction:** Data reader is defined in ```reader.py```and```reader_cv2.py```, Using CV2 reader can improve the speed of reading. In [training stage](#training-a-model-with-flexible-parameters), random crop and flipping are used, while center crop is used in [Evaluation](#evaluation) and [Inference](#inference) stages. Supported data augmentation includes:
* rotation
* color jitter
* random crop
* center crop
* resize
* flipping

## Using Mixed-Precision Training

You may add `--fp16=1` to start train using mixed precisioin training, which the training process will use float16 and the output model ("master" parameters) is saved as float32. You also may need to pass `--scale_loss` to overcome accuracy issues, usually `--scale_loss=8.0` will do.

Note that currently `--fp16` can not use together with `--with_mem_opt`, so pass `--with_mem_opt=0` to disable memory optimization pass.

## Finetuning

Finetuning is to finetune model weights in a specific task by loading pretrained weights. After initializing ```path_to_pretrain_model```, one can finetune a model as:
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

## Evaluation
Evaluation is to evaluate the performance of a trained model. One can download [pretrained models](#supported-models-and-performances) and set its path to ```path_to_pretrain_model```. Then top1/top5 accuracy can be obtained by running the following command:
```
python eval.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=True \
       --pretrained_model=${path_to_pretrain_model}
```

## Inference
Inference is used to get prediction score or image features based on trained models.
```
python infer.py \
       --model=SE_ResNeXt50_32x4d \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=True \
       --pretrained_model=${path_to_pretrain_model}
```

## Supported models and performances

Available top-1/top-5 validation accuracy on ImageNet 2012 are listed in table. Pretrained models can be downloaded by clicking related model names.

- Released models: specify parameter names

|model | top-1/top-5 accuracy(CV2) |
|- |:-: |
|[AlexNet](http://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.tar) | 56.72%/79.17% |
|[VGG11](https://paddle-imagenet-models-name.bj.bcebos.com/VGG11_pretrained.tar) | 69.28%/89.09% |
|[VGG13](https://paddle-imagenet-models-name.bj.bcebos.com/VGG13_pretrained.tar) | 70.02%/89.42% |
|[VGG16](https://paddle-imagenet-models-name.bj.bcebos.com/VGG16_pretrained.tar) | 72.00%/90.69% |
|[VGG19](https://paddle-imagenet-models-name.bj.bcebos.com/VGG19_pretrained.tar) | 72.56%/90.93% |
|[MobileNetV1](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) | 70.99%/89.68% |
|[MobileNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) | 72.15%/90.65% |
|[ResNet18](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar) | 70.98%/89.92% |
|[ResNet34](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar) | 74.57%/92.14% |
|[ResNet50](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar) | 76.50%/93.00% |
|[ResNet50_vc](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vc_pretrained.tar) |78.35%/94.03% |
|[ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar) | 79.12%/94.44% |
|[ResNet101](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) | 77.56%/93.64% |
|[ResNet101_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar) | 79.44%/94.47% |
|[ResNet152](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_pretrained.tar) | 78.26%/93.96% |
|[ResNet152_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_vd_pretrained.tar) | 80.59%/95.30% |
|[ResNet200_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet200_vd_pretrained.tar) | 80.93%/95.33% |
|[ResNeXt101_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_64x4d_pretrained.tar) | 79.35%/94.52% |
|[ResNeXt101_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_vd_64x4d_pretrained.tar) | 80.78%/95.20% |
|[SE_ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt50_32x4d_pretrained.tar) | 78.44%/93.96% |
|[SE_ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt101_32x4d_pretrained.tar) | 79.12%/94.20% |
|[SE154_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SE154_vd_pretrained.tar) | 81.40%/95.48% |
|[GoogleNet](https://paddle-imagenet-models-name.bj.bcebos.com/GoogleNet_pretrained.tar) | 70.70%/89.66% |
|[ShuffleNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar) | 70.03%/89.17% |
|[InceptionV4](https://paddle-imagenet-models-name.bj.bcebos.com/InceptionV4_pretrained.tar) | 80.77%/95.26% |
