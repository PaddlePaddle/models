The minimum PaddlePaddle version needed for the code sample in this directory is the lastest develop branch. If you are on a version of PaddlePaddle earlier than this, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

# SE-ResNeXt for image classification

This model built with paddle fluid is still under active development and is not
the final version. We welcome feedbacks.

## Introduction

The current code support the training of [SE-ResNeXt](https://arxiv.org/abs/1709.01507) (50/152 layers).

## Data Preparation

1. Download ImageNet-2012 dataset
```
mkdir ILSVRC2012/
cd ILSVRC2012/
# get training set
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
# get validation set
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
# prepare directory
tar xf ILSVRC2012_img_train.tar
tar xf ILSVRC2012_img_val.tar
```

2. Generate training and validation file

```train_list.txt``` and ```test_list.txt``` are generated for data provider in ```reader.py```. Lines in the two files are like:
```
ILSVRC2012_val_00000003.JPEG 230
ILSVRC2012_val_00000004.JPEG 809
ILSVRC2012_val_00000005.JPEG 516
ILSVRC2012_val_00000006.JPEG 57
...
```
```
n01440764/n01440764_13575.JPEG 0
n01440764/n01440764_13581.JPEG 0
n01440764/n01440764_13602.JPEG 0
n01440764/n01440764_13625.JPEG 0
...
```
Each line includes an image file path and its label separated by ```SPACE```

## Training a model

To start a training task, one can use command line as:

```
python train.py --num_layers=50 --batch_size=256 --with_mem_opt=True --parallel_exe=True
```
## Inference

The inference process is conducted after each training epoch.

## Results(updating...)

- Top-1/Top-5 Validation Accuracy on ImageNet 2012

The result is obtained by starting with learning rate ```0.1``` and decaying it by ```0.1``` after each ```10``` epoches. The accuracy in table is calculated after ```90``` epoches.

|model | [original paper(Fig.5)](https://arxiv.org/abs/1709.01507) | Pytorch | Paddle fluid
|- | :-: |:-: | -:
|SE-ResNeXt-50 | 77.6%/- | 77.71%/93.63% | 77.42%/93.50%

## Finetune a model

## Released models
|model | Baidu Cloud
|- | -:
|SE-ResNeXt-50 | [url]()
