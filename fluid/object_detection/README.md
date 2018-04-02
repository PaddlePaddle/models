The minimum PaddlePaddle version needed for the code sample in this directory is the lastest develop branch. If you are on a version of PaddlePaddle earlier than this, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

## SSD Object Detection

### Introduction

[Single Shot MultiBox Detector (SSD)](https://arxiv.org/abs/1512.02325) framework for object detection is based on a feed-forward convolutional network. The early network is a standard convolutional architecture for image classification, such as VGG, ResNet, or MobileNet, which is als called base network. In this tutorial we used [MobileNet](https://arxiv.org/abs/1704.04861).

### Data Preparation

You can use [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) or [MS-COCO dataset](http://cocodataset.org/#download).

#### PASCAL VOC Dataset

Download the PASCAL VOC dataset, skip this step if you already have one.

```bash
cd data/
# Download the data.
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# Extract the data.
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```
#### MS-COCO Dataset

```
```
### Train

1. Train on one device (/GPU).

```python
env CUDA_VISIABLE_DEVICES=0 python train.py \
    --paralle=Fale \
    --batch_size=32 --use_gpu=Ture --data='voc'
```

2. Train on multi devices (GPU).

```python
env CUDA_VISIABLE_DEVICES=0,1,2,3 python train.py \
    --paralle=Ture --batch_size=64 \
    --use_gpu=Ture --data='voc'
```

### Evaluate

```python
env CUDA_VISIABLE_DEVICES=0,1,2,3 python eval.py \
    --paralle=Ture --batch_size=64 --use_gpu=Ture \
    --data='voc' --model='model/90'
```

### Infer and Visualize
```python
env CUDA_VISIABLE_DEVICES=0 python infer.py \
    --paralle=False --batch_size=2 \
    --use_gpu=Ture --model='model/90'
```
### Released Model


| Model                 | Pre-trained Model  | Training data    | Test data    | mAP |
|:---------------------:|:------------------:|:----------------:|:------------:|:----:|
|MobileNet-SSD 300x300  | COCO MobileNet SSD | VOC07+12 trainval| VOC07 test   | xx%  |
|MobileNet-SSD 300x300  | ImageNet MobileNet | VOC07+12 trainval| VOC07 test   | xx%  |
|MobileNet-SSD 300x300  | ImageNet MobileNet | MS-COCO trainval | MS-COCO test | xx%  |
