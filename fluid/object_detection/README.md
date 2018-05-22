The minimum PaddlePaddle version needed for the code sample in this directory is the lastest develop branch. If you are on a version of PaddlePaddle earlier than this, [please update your installation](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html).

---

## SSD Object Detection

### Introduction

[Single Shot MultiBox Detector (SSD)](https://arxiv.org/abs/1512.02325) framework for object detection is based on a feed-forward convolutional network. The early network is a standard convolutional architecture for image classification, such as VGG, ResNet, or MobileNet, which is also called base network. In this tutorial we used [MobileNet](https://arxiv.org/abs/1704.04861).

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
  env CUDA_VISIBLE_DEVICES=0 python -u train.py --parallel=False --dataset='pascalvoc' --pretrained_model='pretrained/ssd_mobilenet_v1_coco/'
  ```
  - Train on multi devices (/GPUs).

  ```python
  env CUDA_VISIBLE_DEVICES=0,1 python -u train.py --batch_size=64 --dataset='pascalvoc' --pretrained_model='pretrained/ssd_mobilenet_v1_coco/'
  ```

#### Train on MS-COCO
  - Train on one device (/GPU).
  ```python
  env CUDA_VISIBLE_DEVICES=0 python -u train.py --parallel=False --dataset='coco2014' --pretrained_model='pretrained/mobilenet_imagenet/'
  ```
  - Train on multi devices (/GPUs).
  ```python
  env CUDA_VISIBLE_DEVICES=0,1 python -u train.py --batch_size=64 --dataset='coco2014' --pretrained_model='pretrained/mobilenet_imagenet/'
  ```

TBD

### Evaluate

You can evaluate your trained model in different metric like 11point, integral on both PASCAL VOC and COCO dataset. Moreover, we provide eval_coco_map.py which uses a COCO-specific mAP metric defined by [COCO committee](http://cocodataset.org/#detections-eval). To use this eval_coco_map.py, [cocoapi](https://github.com/cocodataset/cocoapi) is needed.
Install the cocoapi:
```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python2 setup.py install --user
```
Note we set the defualt test list to the dataset's test/val list, you can use your own test list by setting test_list args.

#### Evaluate on PASCAL VOC
```python
env CUDA_VISIBLE_DEVICES=0 python eval.py --dataset='pascalvoc' --model_dir='train_pascal_model/90' --data_dir='data/pascalvoc' --test_list='test.txt' --ap_version='11point'
```

#### Evaluate on MS-COCO
```python
env CUDA_VISIBLE_DEVICES=0 python eval.py --dataset='coco2014' --nms_threshold=0.5 --model_dir='train_coco_model/40' --test_list='annotations/instances_minival2014.json' --ap_version='integral'
env CUDA_VISIBLE_DEVICES=0 python eval_coco_map.py --dataset='coco2017' --nms_threshold=0.5 --model_dir='train_coco_model/40' --test_list='annotations/instances_minival2017.json'
```

TBD

### Infer and Visualize

```python
env CUDA_VISIBLE_DEVICES=0 python infer.py --model_dir='train_coco_model/20' --image_path='./data/coco/val2014/COCO_val2014_000000000139.jpg'
```
Below is the examples after running python infer.py to inference and visualize the model result.
<p align="center">
<img src="images/COCO_val2014_000000000139.jpg" height=300 width=400 hspace='10'/>
<img src="images/COCO_val2014_000000000785.jpg" height=300 width=400 hspace='10'/>
<img src="images/COCO_val2014_000000142324.jpg" height=300 width=400 hspace='10'/>
<img src="images/COCO_val2014_000000144003.jpg" height=300 width=400 hspace='10'/> <br />
MobileNet-SSD300x300 Visualization Examples
</p>

TBD

### Released Model


| Model                    | Pre-trained Model  | Training data    | Test data    | mAP |
|:------------------------:|:------------------:|:----------------:|:------------:|:----:|
|MobileNet-v1-SSD 300x300  | COCO MobileNet SSD | VOC07+12 trainval| VOC07 test   | xx%  |
|MobileNet-v1-SSD 300x300  | ImageNet MobileNet | VOC07+12 trainval| VOC07 test   | xx%  |
|MobileNet-v1-SSD 300x300  | ImageNet MobileNet | MS-COCO trainval | MS-COCO test | xx%  |

TBD
