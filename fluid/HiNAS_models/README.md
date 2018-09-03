# Image Classification Models
This directory contains six image classification models, which are models automatically discovered by Baidu Big Data Lab (BDL) Hierarchical Neural Architecture Search project (HiNAS), achieving 96.1% accuracy on CIFAR-10 dataset. These models are divided into two categories. The first three have no skip link, named HiNAS 0-2, and the last three networks contain skip links, which are similar to the shortcut connections in Resnet, named HiNAS 3-5.

---
## Table of Contents
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training a model](#training-a-model)
- [Model performances](#model-performances)

## Installation
Running the trainer in current directory requires:

- PadddlePaddle Fluid >= v0.15.0
- CuDNN >=6.0

If PaddlePaddle and CuDNN in your runtime environment do not meet the requirements, please follow the instructions in [installation document](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html) and make an update.

## Data preparation

When you run the sample code for the first time, the trainer will automatically download the cifar-10 dataset. Please make sure your environment has an internet connection.

The dataset will be downloaded to `dataset/cifar/cifar-10-python.tar.gz` in the same directory as the Trainer. If automatic download fails, you can go to https://www.cs.toronto.edu/~kriz/cifar.html and download cifar-10-python.tar.gz to the location mentioned above.

## Training a model

After the environment is ready, you can train the model. There are two entrances: `train_hinas.py` and `train_hinas_res.py`. The former is used to train Model 0-2 (without skip link), and the latter is used to train Model 3-5 (contains skip link).

Train Model 0~2 (without skip link)：
```
python train_hinas.py --model=m_id       # m_id can be 0, 1 or 2.
```
Train Model 3~5 (with skip link)：
```
python train_hinas_res.py --model=m_id    # m_id can be 0, 1 or 2.
```

In addition, both `train_hinas.py` and `train_hinas_res.py` support the following parameters:

- **random_flip_left_right**: Random flip image horizontally. (Default: True)
- **random_flip_up_down**: Randomly flip image vertically. (Default: False)
- **cutout**: Add cutout action to image. (Default: True)
- **standardize_image**: Image standardize. (Default: True)
- **pad_and_cut_image**: Random padding image and then crop back to the original size. (Default: True)
- **shuffle_image**: Shuffle the order of the input images during training. (Default: True)
- **lr_max**: Learning rate at the begin of training. (Default: 0.1)
- **lr_min**: Learning rate at the end of training. (Default: 0.0001)
- **batch_size**: Training batch size (Default: 128)
- **num_epochs**: Total training epoch (Default: 200)
- **weight_decay**: L2 Regularization value (Default: 0.0004)
- **momentum**: The momentum parameter in momentum optimizer (Default: 0.9)
- **dropout_rate**: Dropout rate of the dropout layer (Default: 0.5)
- **bn_decay**: The decay/momentum parameter (or called moving average decay) in batch norm layer (Default: 0.9)


## Model performances

Train all six models using same hyperparameters:

- learning rate: 0.1 -> 0.0001 with cosine annealing
- total epoch: 200
- batch size: 128
- L2 decay: 0.000400
- optimizer: momentum optimizer with m=0.9 and use nesterov
- preprocess: random horizontal flip + image standardization + cutout

And below is the accuracy on CIFAR-10 dataset：

| model    | round 1 | round 2 | round 3 | max    | avg    |
|----------|---------|---------|---------|--------|--------|
| HiNAS-0  | 0.9548  | 0.9520  | 0.9513  | 0.9548 | 0.9527 |
| HiNAS-1  | 0.9452  | 0.9462  | 0.9420  | 0.9462 | 0.9445 |
| HiNAS-2  | 0.9508  | 0.9506  | 0.9483  | 0.9508 | 0.9499 |
| HiNAS-3  | 0.9607  | 0.9623  | 0.9601  | 0.9623 | 0.9611 |
| HiNAS-4  | 0.9611  | 0.9584  | 0.9586  | 0.9611 | 0.9594 |
| HiNAS-5  | 0.9578  | 0.9588  | 0.9594  | 0.9594 | 0.9586 |
