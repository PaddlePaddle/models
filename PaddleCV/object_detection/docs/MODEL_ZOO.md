# Model Zoo and Benchmark
## Environment

- Python 2.7.1
- PaddlePaddle 1.5
- CUDA 9.0
- CUDNN 7.4
- NCCL 2.1.2

## Common settings

- All models below except SSD were trained on `coco_2017_train`, and tested on the `coco_2017_val`.
- The Batch Normalization layer of the backbone is replaced by the Affine Channel layer.
- Unless otherwise noted, all ResNet are adopted [ResNet-B](https://arxiv.org/pdf/1812.01187) as the backbone.
- For RCNN and RetinaNet models, only horizontal flipping data augmentation was used in the training phase and no augmentations were used in the testing phase.

## Training Schedules

- We adopt the exactly same training schedules as [Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#training-schedules). 
- 1x indicates the schedule starts at a LR of 0.02 and is decreased by a factor of * 0.1 after 60k and 80k iterations and finally terminates at 90k iterations for minibatch size 16. 
- 2x indicates twice as long as the 1x schedule with the LR change points scaled proportionally.

## ImageNet Pretrained Models

The backbone models pretrained on ImageNet are available. All models are trained on the standard ImageNet-1k dataset. All pretrained models can be download [here](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#supported-models-and-performances)

## Baselines

### Faster & Mask R-CNN

| Backbone             | Type           | Img/gpu | Lr schd | Box AP | Mask AP |                           Download                           |
| :------------------- | :------------- | :-----: | :-----: | :----: | :-----: | :----------------------------------------------------------: |
| ResNet50             | Faster         |    1    |   1x    |  35.1  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection%2Ffaster_rcnn_r50_1x.tar) |
| ResNet50             | Faster         |    1    |   2x    |  37.0  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection%2Ffaster_rcnn_r50_2x.tar) |
| ResNet50             | Mask           |    1    |   1x    |  36.5  |  32.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection%2Fmask_rcnn_r50_1x.tar) |
| ResNet50             | Mask           |    1    |   2x    |        |         |                          [model]()                           |
| ResNet50-D           | Faster         |    1    |   1x    |  36.4  |    -    | [model](ttps://paddlemodels.bj.bcebos.com/object_detection%2Ffaster_rcnn_r50_vd_1x.tar) |
| ResNet50-FPN         | Faster         |    2    |   1x    |  37.2  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection%2Ffaster_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Faster         |    2    |   2x    |  38.1  |    -    | [model](https://paddlemodels.bj.bcebos.com/object_detection%2Ffaster_rcnn_r50_fpn_2x.tar) |
| ResNet50-FPN         | Mask           |    2    |   1x    |  37.9  |  34.2   | [model](https://paddlemodels.bj.bcebos.com/object_detection%2Fmask_rcnn_r50_fpn_1x.tar) |
| ResNet50-FPN         | Mask           |    2    |   2x    |        |         |                          [model]()                           |
| ResNet50-FPN         | Cascade Faster |    2    |   1x    |  40.4  |    -    |                          [model]()                           |
| ResNet50-D-FPN       | Faster         |    2    |   2x    |        |    -    |                          [model]()                           |
| ResNet50-D-FPN       | Mask           |    2    |   2x    |  39.8  |  35.4   | [model](https://paddlemodels.bj.bcebos.com/object_detection%2Fmask_rcnn_r50_vd_fpn_2x.tar) |
| ResNet101            | Faster         |    1    |   1x    |        |    -    |                          [model]()                           |
| ResNet101-FPN        | Faster         |    1    |   1x    |        |    -    |                          [model]()                           |
| ResNet101-FPN        | Faster         |    1    |   2x    |        |    -    |                          [model]()                           |
| ResNet101-FPN        | Mask           |    1    |   1x    |        |         |                          [model]()                           |
| ResNet101-FPN        | Mask           |    1    |   2x    |        |         |                          [model]()                           |
| ResNet101-D-FPN      | Faster         |    1    |   1x    |        |    -    |                          [model]()                           |
| ResNet101-D-FPN      | Faster         |    1    |   2x    |        |    -    |                          [model]()                           |
| ResNet101-D-FPN      | Mask           |    1    |   2x    |        |         |                          [model]()                           |
| ResNeXt101-64x4d-FPN | Faster         |    1    |   1x    |        |    -    |                          [model]()                           |
| ResNeXt101-64x4d-FPN | Faster         |    1    |   2x    |        |    -    |                          [model]()                           |
| ResNeXt101-64x4d-FPN | Mask           |    1    |   1x    |        |         |                          [model]()                           |
| ResNeXt101-64x4d-FPN | Mask           |    1    |   2x    |        |         |                          [model]()                           |
| SENet154-D-FPN       | Faster         |    1    |  1.44x  |        |    -    |                          [model]()                           |
| SENet154-D-FPN       | Mask           |    1    |  1.44x  |  44.0  |  38.7   | [model](https://paddlemodels.bj.bcebos.com/object_detection%2Fmask_rcnn_se154_vd_fpn_s1x.tar) |

### Yolo v3

| Backbone  | Size | Lr schd | Box AP | Download  |
| :-------- | :--: | :-----: | :----: | :-------: |
| DarkNet53 | 608  |  120e   |  25.7  | [model]() |

- Notes: Data Augmentation（TODO：Kaipeng）

### RetinaNet

| Backbone     | Size | Lr schd | Box AP | Download  |
| :----------- | :--: | :-----: | :----: | :-------: |
| ResNet50-FPN | 300  |  120e   |  25.7  | [model]() |

- Notes: （TODO：Kaipeng）

### SSD

| Backbone     | Size | Lr schd | Box AP | Download  |
| :----------- | :--: | :-----: | :----: | :-------: |
| MobileNet v1 | 300  |  120e   |  25.7  | [model]() |

