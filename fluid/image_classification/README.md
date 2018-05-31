The minimum PaddlePaddle version needed for the code sample in this directory is the lastest develop branch. If you are on a version of PaddlePaddle earlier than this, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

# SE-ResNeXt for image classification

This model built with paddle fluid is still under active development and is not
ang
he final version. We welcome feedbacks.

## Introduction

The current code support the training of [SE-ResNeXt](https://arxiv.org/abs/1709.01507) (50/152 layers).

## Data Preparation

1. Download ImageNet-2012 dataset
```
cd data/
mkdir -p ILSVRC2012/
cd ILSVRC2012/
# get training set
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
# get validation set
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
# prepare directory
tar xf ILSVRC2012_img_train.tar
tar xf ILSVRC2012_img_val.tar

# unzip all classes data using unzip.sh
sh unzip.sh
```

2. Download training and validation label files from [ImageNet2012 url](https://pan.baidu.com/s/1Y6BCo0nmxsm_FsEqmx2hKQ)(password:```wx99```). Untar it into workspace ```ILSVRC2012/```. The files include

**train_list.txt**: training list of imagenet 2012 classification task, with each line seperated by SPACE.
```
train/n02483708/n02483708_2436.jpeg 369
train/n03998194/n03998194_7015.jpeg 741
train/n04523525/n04523525_38118.jpeg 884
train/n04596742/n04596742_3032.jpeg 909
train/n03208938/n03208938_7065.jpeg 535
...
```
**val_list.txt**: validation list of imagenet 2012 classification task, with each line seperated by SPACE.
```
val/ILSVRC2012_val_00000001.jpeg 65
val/ILSVRC2012_val_00000002.jpeg 970
val/ILSVRC2012_val_00000003.jpeg 230
val/ILSVRC2012_val_00000004.jpeg 809
val/ILSVRC2012_val_00000005.jpeg 516
...
```
**synset_words.txt**: the semantic label of each class.

## Training a model

To start a training task, one can use command line as:

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

## Results

The SE-ResNeXt-50 model is trained by starting with learning rate ```0.1``` and decaying it by ```0.1``` after each ```10``` epoches. Top-1/Top-5 Validation Accuracy on ImageNet 2012 is listed in table.

|model | [original paper(Fig.5)](https://arxiv.org/abs/1709.01507) | Pytorch | Paddle fluid
|- | :-: |:-: | -:
|SE-ResNeXt-50 | 77.6%/- | 77.71%/93.63% | 77.42%/93.50%


## Released models
|model | Baidu Cloud
|- | -:
|SE-ResNeXt-50 | [url]()
TBD
