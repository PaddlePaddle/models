# Image Classification and Paddle Model Zoo
This page introduces how to do image classification with Paddle fluid. To run the examples below, please [install](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html) the newest Paddle first.

---

To train a model using ImageNet dataset, please follow the steps below.


## Data Preparation

1. Download ImageNet-2012 dataset from website
```
cd data/
mkdir -p ILSVRC2012/
cd ILSVRC2012/
wget paddl_imagenet2012_dataset_url/ImageNet2012_dataset.tar
tar xf ImageNet2012_dataset.tar
```

2. Download training and validation label files
```
wget paddl_imagenet2012_label_url/ImageNet2012_label.tar
tar xf ImageNet2012_label.tar
```
there are two label files which contain train and validation image labels respectively:

**train_list.txt**: label file imagenet-2012 training set, with each line seperated by SPACE, like:
```
train/n02483708/n02483708_2436.jpeg 369
train/n03998194/n03998194_7015.jpeg 741
train/n04523525/n04523525_38118.jpeg 884
train/n04596742/n04596742_3032.jpeg 909
train/n03208938/n03208938_7065.jpeg 535
...
```
**val_list.txt**: label file of imagenet-2012 validation set, with each line seperated by SPACE, like.
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
       --model=SE_ResNeXt101_32x4d \
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

## Finetuning
```
python train.py
       --model=SE_ResNeXt101_32x4d \
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

Models are trained by starting with learning rate ```0.1``` and decaying it by ```0.1``` after each ```30``` epoches, if not special introduced. Available top-1/top-5 validation accuracy on ImageNet 2012 is listed in table.

|model | top-1/top-5 accuracy
|- | -:
|AlexNet | -
|VGG11 | -
|VGG13 | -
|VGG16 | -
|VGG19 | -
|GoogleNet | -
|InceptionV4 | -
|MobileNet | -
|ResNet50 | -
|ResNet101 | -
|ResNet152 | -
|SE_ResNeXt50_32x4d | 77.42%/93.50%
|SE_ResNeXt101_32x4d | -
|SE_ResNeXt152_32x4d | -
|DPN68 | -
|DPN92 | -
|DPN98 | -
|DPN107 | -
|DPN131 | -


## Download models
|model | url
|- | -:
|SE-ResNeXt-50 | [url]()
TBD
