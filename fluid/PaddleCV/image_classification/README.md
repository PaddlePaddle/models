# Image Classification and Model Zoo
Image classification, which is an important field of computer vision, is to classify an image into pre-defined labels. Recently, many researchers developed different kinds of neural networks and highly improve the classification performance. This page introduces how to do image classification with PaddlePaddle Fluid, including [data preparation](#data-preparation), [training](#training-a-model), [finetuning](#finetuning), [evaluation](#evaluation) and [inference](#inference).

---
## Table of Contents
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training a model with flexible parameters](#training-a-model)
- [Using Mixed-Precision Training](#using-mixed-precision-training)
- [Finetuning](#finetuning)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Supported models and performances](#supported-models)

## Installation

Running sample code in this directory requires PaddelPaddle Fluid v0.13.0 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in [installation document](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html) and make an update.

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
train/n04596742/n04596742_3032.jpeg 909
train/n03208938/n03208938_7065.jpeg 535
...
```
* *val_list.txt*: label file of imagenet-2012 validation set, with each line seperated by ```SPACE```, like.
```
val/ILSVRC2012_val_00000001.jpeg 65
val/ILSVRC2012_val_00000002.jpeg 970
val/ILSVRC2012_val_00000003.jpeg 230
val/ILSVRC2012_val_00000004.jpeg 809
val/ILSVRC2012_val_00000005.jpeg 516
...
```

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
* **lr_strategy**: learning rate changing strategy. Default: "piecewise_decay".
* **lr**: initialized learning rate. Default: 0.1.
* **pretrained_model**: model path for pretraining. Default: None.
* **checkpoint**: the checkpoint path to resume. Default: None.
* **model_category**: the category of models, ("models"|"models_name"). Default: "models".

Or can start the training step by running the ```run.sh```.

**data reader introduction:** Data reader is defined in ```reader.py``` and ```reader_cv2.py```, Using CV2 reader can improve the speed of reading. In [training stage](#training-a-model), random crop and flipping are used, while center crop is used in [evaluation](#inference) and [inference](#inference) stages. Supported data augmentation includes:
* rotation
* color jitter
* random crop
* center crop
* resize
* flipping

**training curve:** The training curve can be drawn based on training log. For example, the log from training AlexNet is like:
```
End pass 1, train_loss 6.23153877258, train_acc1 0.0150696625933, train_acc5 0.0552518665791, test_loss 5.41981744766, test_acc1 0.0519132651389, test_acc5 0.156150355935
End pass 2, train_loss 5.15442800522, train_acc1 0.0784279331565, train_acc5 0.211050540209, test_loss 4.45795249939, test_acc1 0.140469551086, test_acc5 0.333163291216
End pass 3, train_loss 4.51505613327, train_acc1 0.145300447941, train_acc5 0.331567406654, test_loss 3.86548018456, test_acc1 0.219443559647, test_acc5 0.446448504925
End pass 4, train_loss 4.12735557556, train_acc1 0.19437250495, train_acc5 0.405713528395, test_loss 3.56990146637, test_acc1 0.264536827803, test_acc5 0.507190704346
End pass 5, train_loss 3.87505435944, train_acc1 0.229518383741, train_acc5 0.453582793474, test_loss 3.35345435143, test_acc1 0.297349333763, test_acc5 0.54753267765
End pass 6, train_loss 3.6929500103, train_acc1 0.255628824234, train_acc5 0.487188398838, test_loss 3.17112898827, test_acc1 0.326953113079, test_acc5 0.581780135632
End pass 7, train_loss 3.55882954597, train_acc1 0.275381118059, train_acc5 0.511990904808, test_loss 3.03736782074, test_acc1 0.349035382271, test_acc5 0.606293857098
End pass 8, train_loss 3.45595097542, train_acc1 0.291462600231, train_acc5 0.530815005302, test_loss 2.96034455299, test_acc1 0.362228929996, test_acc5 0.617390751839
End pass 9, train_loss 3.3745200634, train_acc1 0.303871691227, train_acc5 0.545210540295, test_loss 2.93932366371, test_acc1 0.37129303813, test_acc5 0.623573005199
...
```

The error rate curves of AlexNet, ResNet50 and SE-ResNeXt-50 are shown in the figure below.
<p align="center">
<img src="images/curve.jpg" height=480 width=640 hspace='10'/> <br />
Training and validation Curves
</p>


## Using Mixed-Precision Training

You may add `--fp16 1` to start train using mixed precisioin training, which the training process will use float16 and the output model ("master" parameters) is saved as float32. You also may need to pass `--scale_loss` to overcome accuracy issues, usually `--scale_loss 8.0` will do.

Note that currently `--fp16` can not use together with `--with_mem_opt`, so pass `--with_mem_opt 0` to disable memory optimization pass.

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
Evaluation is to evaluate the performance of a trained model. One can download [pretrained models](#supported-models) and set its path to ```path_to_pretrain_model```. Then top1/top5 accuracy can be obtained by running the following command:
```
python eval.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=True \
       --pretrained_model=${path_to_pretrain_model}
```

According to the congfiguration of evaluation, the output log is like:
```
Testbatch 0,loss 2.1786134243, acc1 0.625,acc5 0.8125,time 0.48 sec
Testbatch 10,loss 0.898496925831, acc1 0.75,acc5 0.9375,time 0.51 sec
Testbatch 20,loss 1.32524681091, acc1 0.6875,acc5 0.9375,time 0.37 sec
Testbatch 30,loss 1.46830511093, acc1 0.5,acc5 0.9375,time 0.51 sec
Testbatch 40,loss 1.12802267075, acc1 0.625,acc5 0.9375,time 0.35 sec
Testbatch 50,loss 0.881597697735, acc1 0.8125,acc5 1.0,time 0.32 sec
Testbatch 60,loss 0.300163716078, acc1 0.875,acc5 1.0,time 0.48 sec
Testbatch 70,loss 0.692037761211, acc1 0.875,acc5 1.0,time 0.35 sec
Testbatch 80,loss 0.0969972759485, acc1 1.0,acc5 1.0,time 0.41 sec
...
```

## Inference
Inference is used to get prediction score or image features based on trained models.
```
python infer.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=True \
       --pretrained_model=${path_to_pretrain_model}
```
The output contains predication results, including maximum score (before softmax) and corresponding predicted label.
```
Test-0-score: [13.168352], class [491]
Test-1-score: [7.913302], class [975]
Test-2-score: [16.959702], class [21]
Test-3-score: [14.197695], class [383]
Test-4-score: [12.607652], class [878]
Test-5-score: [17.725458], class [15]
Test-6-score: [12.678599], class [118]
Test-7-score: [12.353498], class [505]
Test-8-score: [20.828007], class [747]
Test-9-score: [15.135801], class [315]
Test-10-score: [14.585114], class [920]
Test-11-score: [13.739927], class [679]
Test-12-score: [15.040644], class [386]
...
```

## Supported models and performances

Models consists of two categories: Models with specified parameters names in model definition and Models without specified parameters, Generate named model by indicating ```model_category = models_name```.

Models are trained by starting with learning rate ```0.1``` and decaying it by ```0.1``` after each pre-defined epoches, if not special introduced. Available top-1/top-5 validation accuracy on ImageNet 2012 are listed in table. Pretrained models can be downloaded by clicking related model names.


- Released models: specify parameter names

|model | top-1/top-5 accuracy(PIL)| top-1/top-5 accuracy(CV2) |
|- |:-: |:-:|
|[AlexNet](http://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.zip) | 56.71%/79.18% | 55.88%/78.65% |
|[VGG11](https://paddle-imagenet-models-name.bj.bcebos.com/VGG11_pretrained.zip) | 69.22%/89.09% | 69.01%/88.90% |
|[VGG13](https://paddle-imagenet-models-name.bj.bcebos.com/VGG13_pretrained.zip) | 70.14%/89.48% | 69.83%/89.13% |
|[VGG16](https://paddle-imagenet-models-name.bj.bcebos.com/VGG16_pretrained.zip) | 72.08%/90.63% | 71.65%/90.57% |
|[VGG19](https://paddle-imagenet-models-name.bj.bcebos.com/VGG19_pretrained.zip) | 72.56%/90.83% | 72.32%/90.98% |
|[MobileNetV1](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.zip) | 70.91%/89.54% | 70.51%/89.35% |
|[MobileNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.zip) | 71.90%/90.55% | 71.53%/90.41% |
|[ResNet50](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.zip) | 76.35%/92.80% | 76.22%/92.92% |
|[ResNet101](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.zip) | 77.49%/93.57% | 77.56%/93.64% |
|[ResNet152](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_pretrained.zip) | 78.12%/93.93% | 77.92%/93.87% |
|[SE_ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNext50_32x4d_pretrained.zip) | 78.50%/94.01% | 78.44%/93.96% |
|[SE_ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt101_32x4d_pretrained.zip) | 79.26%/94.22% | 79.12%/94.20% |




- Released models: not specify parameter names

|model | top-1/top-5 accuracy(PIL)| top-1/top-5 accuracy(CV2) |
|- |:-: |:-:|
|[ResNet152](http://paddle-imagenet-models.bj.bcebos.com/ResNet152_pretrained.zip) | 78.18%/93.93% | 78.11%/94.04% |
|[SE_ResNeXt50_32x4d](http://paddle-imagenet-models.bj.bcebos.com/se_resnext_50_model.tar) | 78.32%/93.96% | 77.58%/93.73% |
