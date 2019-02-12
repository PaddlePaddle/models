# LRC Local Rademachar Complexity Regularization
This directory contains image classification model based on novel regularizer rooted in Local Rademacher Complexity (LRC). The regularization by LRC and [DARTS](https://arxiv.org/abs/1806.09055) are combined in this model and it achieves 97.3% accuracy on CIFAR-10 dataset.

---
# Table of Contents

- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Model performances](#model-performances)

## Installation

Running sample code in this directory requires PaddelPaddle Fluid v.1.2.0 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in [installation document](http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/install/index_cn.html#paddlepaddle) and make an update.

## Data preparation

When you want to use the cifar-10 dataset for the first time, you can download the dataset as:

    sh ./dataset/download.sh

Please make sure your environment has an internet connection.

The dataset will be downloaded to `dataset/cifar/cifar-10-batches-py` in the same directory as the `train.py`. If automatic download fails, you can download cifar-10-python.tar.gz from https://www.cs.toronto.edu/~kriz/cifar.html and decompress it to the location mentioned above.


## Training

After data preparation, one can start the training step by:

    python -u train_mixup.py \
        --batch_size=80 \
        --auxiliary \
        --weight_decay=0.0003 \
        --learning_rate=0.025 \
        --lrc_loss_lambda=0.7 \
        --cutout
- Set ```export CUDA_VISIBLE_DEVICES=0``` to specifiy one GPU to train.
- For more help on arguments:

    python train_mixup.py --help

**data reader introduction:**

* Data reader is defined in `reader.py`.
* Reshape the images to 32 * 32.
* In training stage, images are padding to 40 * 40 and cropped randomly to the original size.
* In training stage, images are horizontally random flipped.
* Images are standardized to (0, 1).
* In training stage, cutout images randomly.
* Shuffle the order of the input images during training.

**model configuration:**

* Use auxiliary loss and auxiliary\_weight=0.4.
* Use dropout and drop\_path\_prob=0.2.
* Set lrc\_loss\_lambda=0.7.

**training strategy:**

*  Use momentum optimizer with momentum=0.9.
*  Weight decay is 0.0003.
*  Use cosine decay with init\_lr=0.025.
*  Total epoch is 600.
*  Use Xaiver initalizer to weight in conv2d, Constant initalizer to weight in batch norm and Normal initalizer to weight in fc.
*  Initalize bias in batch norm and fc to zero constant and do not add bias to conv2d.


## Model performances
Below is the accuracy on CIFAR-10 dataset：

| model | avg top1 | avg top5 |
| ----- | -------- | -------- |
| [DARTS-LRC](https://paddlemodels.bj.bcebos.com/autodl/fluid_rademacher.tar.gz) | 97.34 | 99.75 |
