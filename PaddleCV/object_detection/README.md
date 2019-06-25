# PaddlePaddle Object Detection

This object detection framework is based on PaddlePaddle. We want to provide classical and state of the art detection algorithms in generic object detection and specific target detection for uers. And we aimed to make this framework easy to extend, train and depoly uers' task. We aimed to make it easy to use in research and products.

## Introduction

Major features:

- Easy to Depoly:
  All the operations related with inference are implemented by C++ and CUDA, it makes the detection model easy to depoly in products on server without Python based on the high efficient inference engine of PaddlePaddle.
  We release detection models based on ResNet-vd backbone, for example, the accuracy of Faster RCNN model with FPN based on ResNet50 VD is close to model based on ResNet 101. But the former is more smaller and faster.

- Easy to Customize:
   All components are modular encapsulated, including the data transforms. It's easy to plug in and pull out any module. For example, uers can switch backone easily or add data mixed-up data transform for models.

- High Efficiency:
  Based on high efficient PaddlePaddle framework, less memory is required. For example, the batch size of Mask-RCNN based on ResNet50 can be 5 per Tesla V100 (16G). The training speed of Yolo v3 is faster than other framework.  



The supported architectures are as follows:


|                    | ResNet |ResNet vd| ResNeXt  | SENet    | MobileNet | DarkNet|
|--------------------|:------:|--------:|:--------:|:--------:|:---------:|:------:|
| Faster R-CNN       | ✓      | ✓      | ✓        |  ✓       | ✗        | ✗      |
| Faster R-CNN + FPN | ✓      | ✓      | ✓        |  ✓       | ✗        | ✗      |
| Mask R-CNN         | ✓      | ✓      | ✓        |  ✓       | ✗        | ✗      |
| Mask R-CNN + FPN   | ✓      | ✓      | ✓        |  ✓       | ✗        | ✗      |
| Cascade R-CNN      | ✓      | ✓      | ✓        |  ✓       | ✗        | ✗      |
| RetinaNet          | ✓      | ✓      | ✓        |  ✓       | ✗        | ✗      |
| Yolov3             | ✓      | ✗      | ✗         |  ✗       | ✓        | ✓     |
| SSD                | ✗      | ✗      | ✗         |  ✗       | ✓        | ✗      |

The extensive capabilities:

- [x] **Synchronized batch norm**:  used in Yolo v3.
- [x] **Group Norm**: supported this operation, the related model will be added later.
- [x] **Dodulated deformable convolution**: suppored this operation, the related model will be added later.
- [x] **Deformable PSRoI Pooling**: suppored this operation, the related model will be added later.


#### Work in Progress and to Do

- About Framework:
   - Mixed precision training and distributed training.
   - 8-bit deployment.
   - Easy to customize user-defined function.

- About Algorithms:
   - More SOTA models.
   - More easy-to-deployed models.


We are also pleased to receive your feedback.

## Model zoo

The trained models can be available in PaddlePaddle [detection model zoo](docs/MODEL_ZOO.md).

## Installation

Please follow the [installation instructions](docs/INSTALL.md) to install PaddlePaddle and prepare environment.

## Get Started

For quickly start, infer a image:

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python tools/infer.py -c configs/mask_rcnn_r50_1x.yml \
    -o weights=http://
    -
```

The predicted result is visualized in `output/xxx.jpg`.

For more detailed training and evaluating pipeline, please refer [GETTING_STARTED.md](docs/GETTING_STARTED.md).


For more documentation, please refer:

- [How to config an object detection pipeline.](docs/CONFIG.md)
- [How to use customized dataset and add data preprocessing.](docs/DATA.md)


## Deploy

The example how to use PaddlePaddle to deploy detection model will be added later.


## Framework

The code tree is as follows.

```
|-- configs               # all model configs
|-- demo                  # demo
|-- docs                  # documents
|-- ppdet  
|   |-- core              #  module registration and config parsing
|   |-- data  
|   |   |-- source        # dataset parsing
|   |   |-- tools         # data conversion tools for custom dataset
|   |   `-- transform     # data transform and parallel processing
|   |-- modeling          # core modeling modules
|   |   |-- anchor_heads  
|   |   |-- architectures
|   |   |-- backbones
|   |   |-- roi_extractors
|   |   |-- roi_heads
|   `-- utils             # common components.
`-- tools                 # running interface，like trainning, config generating.
```



## Updates

The major updates are as follows:

#### 2019-07-03
- Release the unified detection framework.
- Supported algorithms: Faster R-CNN, Mask R-CNN, Faster R-CNN + FPN, Mask R-CNN + FPN, Cascade-Faster-RCNN + FPN, RetinaNet, Yolo v3 and SSD.
- Release first version of model zoo.


## Contributing

We appreciate everyone's contributions!
