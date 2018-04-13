The minimum PaddlePaddle version needed for the code sample in this directory is the lastest develop branch. If you are on a version of PaddlePaddle earlier than this, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

## SSD Object Detection

### Introduction

[Single Shot MultiBox Detector (SSD)](https://arxiv.org/abs/1512.02325) framework for object detection is based on a feed-forward convolutional network. The early network is a standard convolutional architecture for image classification, such as VGG, ResNet, or MobileNet, which is als called base network. In this tutorial we used [MobileNet](https://arxiv.org/abs/1704.04861).

### Data Preparation

You can use [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) or [MS-COCO dataset](http://cocodataset.org/#download).

#### PASCAL VOC Dataset

If you want to train model on PASCAL VOC dataset, please download datset at first, skip this step if you already have one.

```bash
cd data/pascalvoc
./download.sh
```

The command `download.sh` also will create training and testing file lists.

#### MS-COCO Dataset

If you want to train model on MS-COCO dataset, please download datset at first, skip this step if you already have one.

```
cd data/coco
./download.sh
```

### Train

#### Download the Pre-trained Model.

We provide two pre-trained models. The one is MobileNet-v1 SSD trained on COCO dataset, but removed the convolutional predictors for COCO dataset. This model can be used to initialize the models when training other dataset, like PASCAL VOC. Then other pre-trained model is MobileNet v1 trained on ImageNet 2012 dataset, but removed the last weights and bias in Fully-Connected layer.

Declaration: the MobileNet-v1 SSD model is converted by [TensorFlow model](https://github.com/tensorflow/models/blob/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/object_detection/g3doc/detection_model_zoo.md). The MobileNet v1 model is converted [Caffe](https://github.com/shicai/MobileNet-Caffe).

  - Download MobileNet-v1 SSD:
    ```
    ./pretrained/download_coco.sh
    ```
  - Download MobileNet-v1:
    ```
    ./pretrained/download_imagenet.sh
    ```

#### Train on PASCAL VOC
  - Train on one device (/GPU).
  ```python
  env CUDA_VISIABLE_DEVICES=0 python -u train.py --parallel=False --data='pascalvoc' --pretrained_model='pretrained/ssd_mobilenet_v1_coco/'
  ```
  - Train on multi devices (/GPUs).

  ```python
  env CUDA_VISIABLE_DEVICES=0,1 python -u train.py --batch_size=64 --data='pascalvoc' --pretrained_model='pretrained/ssd_mobilenet_v1_coco/'
  ```

#### Train on MS-COCO
  - Train on one device (/GPU).
  ```python
  env CUDA_VISIABLE_DEVICES=0 python -u train.py --parallel=False --data='coco' --pretrained_model='pretrained/mobilenet_imagenet/'
  ```
  - Train on multi devices (/GPUs).
  ```python
  env CUDA_VISIABLE_DEVICES=0,1 python -u train.py --batch_size=64 --data='coco' --pretrained_model='pretrained/mobilenet_imagenet/'
  ```

TBD

### Evaluate

```python
env CUDA_VISIABLE_DEVICES=0 python eval.py --model='model/90' --test_list=''
```

TBD

### Infer and Visualize

```python
env CUDA_VISIABLE_DEVICES=0 python infer.py --batch_size=2 --model='model/90' --test_list=''
```

TBD

### Released Model


| Model                    | Pre-trained Model  | Training data    | Test data    | mAP |
|:------------------------:|:------------------:|:----------------:|:------------:|:----:|
|MobileNet-v1-SSD 300x300  | COCO MobileNet SSD | VOC07+12 trainval| VOC07 test   | xx%  |
|MobileNet-v1-SSD 300x300  | ImageNet MobileNet | VOC07+12 trainval| VOC07 test   | xx%  |
|MobileNet-v1-SSD 300x300  | ImageNet MobileNet | MS-COCO trainval | MS-COCO test | xx%  |

TBD
