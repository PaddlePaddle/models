# Simple Baselines for Human Pose Estimation in Fluid

## Introduction
This is a simple demonstration of re-implementation in [PaddlePaddle.Fluid](http://www.paddlepaddle.org/en) for the paper [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208) (ECCV'18) from MSRA.

![demo](demo.gif)

> **Video in Demo**: *Bruno Mars - That’s What I Like [Official Video]*.

We also recommend users to take a look at the [IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/122271)

## Requirements

  - Python == 2.7 or 3.6
  - PaddlePaddle >= 1.1.0
  - opencv-python >= 3.3

### Notes:
We found that there are some issues may result in misconvergence with PaddlePaddle 1.3.0 and cuDNN-7.0. So it is recommended to use the latest version of PaddlePaddle (>= 1.4).

## Environment

The code is developed and tested under 4 Tesla K40/P40 GPUS cards on CentOS with installed CUDA-9.0/8.0 and cuDNN-7.0.

## Results on MPII Val
| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1| Models |
| ---- |:----:|:--------:|:-----:|:-----:|:---:|:----:|:-----:|:----:|:-------:|:------:|
| 256x256\_pose\_resnet\_50 in PyTorch | 96.351    | 95.329 | 88.989 | 83.176 | 88.420    | 83.960 | 79.594 | 88.532 | 33.911 | - |
| 256x256\_pose\_resnet\_50 in Fluid   | 96.385 | 95.363 | 89.211 | 84.084 | 88.454 | 84.182 | 79.546 | 88.748 | 33.750 | [`link`](https://paddlemodels.bj.bcebos.com/pose/pose-resnet50-mpii-256x256.tar.gz) |
| 384x384\_pose\_resnet\_50 in PyTorch | 96.658 | 95.754 | 89.790 | 84.614 | 88.523 | 84.666 | 79.287 | 89.066 | 38.046 | - |
| 384x384\_pose\_resnet\_50 in Fluid   | 96.862 | 95.635 | 90.046 | 85.557 | 88.818 | 84.948 | 78.484 | 89.235 | 38.093 | [`link`](https://paddlemodels.bj.bcebos.com/pose/pose-resnet50-mpii-384x384.tar.gz) |

## Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) | Models |
| ---- |:--:|:-----:|:------:|:------:|:------:|:--:|:-----:|:------:|:------:|:------:|:------:|
| 256x192\_pose\_resnet\_50 in PyTorch | 0.704 | 0.886 | 0.783 | 0.671 | 0.772 | 0.763 | 0.929 | 0.834 | 0.721 | 0.824 | - |
| 256x192\_pose\_resnet\_50 in Fluid   | 0.712 | 0.897 | 0.786 | 0.683 | 0.756 | 0.741 | 0.906 | 0.806 | 0.709 | 0.790 | [`link`](https://paddlemodels.bj.bcebos.com/pose/pose-resnet50-coco-256x192.tar.gz) |
| 384x288\_pose\_resnet\_50 in PyTorch | 0.722 | 0.893 | 0.789 | 0.681 | 0.797 | 0.776 | 0.932 | 0.838 | 0.728 | 0.846 | - |
| 384x288\_pose\_resnet\_50 in Fluid   | 0.727 | 0.897 | 0.796 | 0.690 | 0.783 | 0.754 | 0.907 | 0.813 | 0.714 | 0.814 | [`link`](https://paddlemodels.bj.bcebos.com/pose/pose-resnet50-coco-384x288.tar.gz) |

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

Then, put them in the folder `pretrained` under the directory root of this repo, make them look like:

```
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

Downloading the checkpoints of Pose-ResNet-50 trained on MPII dataset from [here](https://paddlemodels.bj.bcebos.com/pose/pose-resnet50-mpii-384x384.tar.gz). Extract it into the folder `checkpoints` under the directory root of this repo. Then run

```bash
python val.py --dataset mpii --checkpoint checkpoints/pose-resnet50-mpii-384x384 --data_root data/mpii
```

### Perform Training

```bash
python train.py --dataset mpii
```

**Note**: Configurations for training are aggregated in the `lib/mpii_reader.py` and `lib/coco_reader.py`.

### Perform Test on Images

We also support to apply pre-trained models on customized images.

Put the images into the folder `test` under the directory root of this repo. Then run

```bash
# default is MPII dataset
python test.py --checkpoint checkpoints/pose-resnet-50-384x384-mpii
```

`python test.py --help` for more usage.

If there are multiple persons in images, detectors such as [Faster R-CNN](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/rcnn), [SSD](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/object_detection) or others should be used first to crop them out. Because the simple baseline for human pose estimation is a top-down method.

## Reference

  - Simple Baselines for Human Pose Estimation and Tracking in PyTorch [`code`](https://github.com/Microsoft/human-pose-estimation.pytorch#data-preparation)

## License

This code is released under the Apache License 2.0.
