# Simple Baselines for Human Pose Estimation in Fluid

## Introduction
This is a simple demonstration of re-implementation in [PaddlePaddle.Fluid](http://www.paddlepaddle.org/en) for the paper [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208) (ECCV'18) from MSRA. 

![demo](demo.gif)

> **Video in Demo**: *Bruno Mars - That’s What I Like [Official Video]*.

## Requirements

- Python == 2.7
- PaddlePaddle >= 1.0
- opencv-python >= 3.3
- tqdm >= 4.25

## Environment

The code is developed and tested under 4 Tesla K40 GPUS cards on CentOS with installed CUDA-9.2/8.0 and cuDNN-7.1.

## Known Issues

- The model does not converge with large batch\_size (e.g. = 32) on Tesla P40 / V100 / P100 GPUS cards, because PaddlePaddle uses the batch normalization function of cuDNN. Changing batch\_size into 1 image on each card during training will ease this problem, but not sure the performance. The issue can be tracked at [here](https://github.com/PaddlePaddle/Paddle/issues/14580).

## Results on MPII Val
| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1| Models |
| ---- |:----:|:--------:|:-----:|:-----:|:---:|:----:|:-----:|:----:|:-------:|:------:|
| 383x384\_pose\_resnet\_50 in PyTorch | 96.658 | 95.754 | 89.790 | 84.614 | 88.523 | 84.666 | 79.287 | 89.066 | 38.046 | - |
| 383x384\_pose\_resnet\_50 in Fluid   | 96.248 | 95.346 | 89.807 | 84.873 | 88.298 | 83.679 | 78.649 | 88.767 | 37.374 | [`link`](tbd) |

### Notes:

- Flip test is used.
- We do not hardly search the best model, just use the last saved model to make validation.

## Getting Start

### Prepare Datasets and Pretrained Models

- Following the [instruction](https://github.com/Microsoft/human-pose-estimation.pytorch#data-preparation) to prepare datasets.
- Download the pretrained ResNet-50 model in PaddlePaddle.Fluid on ImageNet from [Model Zoo](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification#supported-models-and-performances). 

```bash
wget http://paddle-imagenet-models.bj.bcebos.com/resnet_50_model.tar
```

Put it in the folder `pretrained` under the directory root of this repo, make it like

```bash
${THIS REPO ROOT}
 `-- pretrained
     `-- resnet_50
     	|-- 115
 `-- data
 	 `-- coco
 	 	|-- annotations
 	 	|-- images
 	 `-- mpii
 	 	|-- annot
 	 	|-- images
 ...
```

### Install [COCOAPI](https://github.com/cocodataset/cocoapi)

```bash
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# if cython is not installed
pip install Cython
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python2 setup.py install --user
```

### Perform Validating

Downloading the checkpoints of Pose-ResNet-50 trained on MPII dataset from [here](tbd). Extract it into the folder `checkpoints` under the directory root of this repo. Then run

```bash
python2 val.py --dataset 'mpii' --checkpoint 'checkpoints/pose-resnet-50-384x384-mpii'
```

### Perform Training

```bash
python2 train.py --dataset 'mpii' # or coco
```

### Perform Test on Images

Put the images into the folder `test` under the directory root of this repo. Then run

```bash
python2 test.py --checkpoint 'checkpoints/pose-resnet-50-384x384-mpii'
```

If there are multiple persons in images, detectors such as [Faster R-CNN](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/faster_rcnn), [SSD](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/object_detection) or others should be used first to crop them out. Because the simple baseline for human pose estimation is a top-down method.

## Reference

- Simple Baselines for Human Pose Estimation and Tracking in PyTorch [`code`](https://github.com/Microsoft/human-pose-estimation.pytorch#data-preparation)

## License

This code is released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.
