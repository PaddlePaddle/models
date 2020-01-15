# SemSegPaddle: A Paddle-based Framework for Deep Learning in Semantic Segmentation

This is a Paddle implementation of semantic segmentation models on multiple datasets, including Cityscapes, Pascal Context, and ADE20K.

## Updates

- [**2020/01/08**] We release ***PSPNet-ResNet101*** and ***GloRe-ResNet101*** models on Pascal Context and Cityscapes datasets.

## Highlights

Synchronized Batch Normlization is important for segmenation.
  - The implementation is easy to use as it is pure-python, no any C++ extra extension libs.
   
  - Paddle provides sync_batch_norm.
   
   
## Support models

We split our models into backbone and decoder network, where backbone network are transfered from classification networks.

Backbone:
  - ResNet
  - ResNeXt
  - HRNet
  - EfficientNet
  
Decoder:
  - PSPNet: [Pyramid Scene Parsing Network](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf)
  - DeepLabv3: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
  - GloRe: [Graph-Based Global Reasoning Networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Graph-Based_Global_Reasoning_Networks_CVPR_2019_paper.pdf)
  - GINet: [GINet: Graph Interaction Netowrk for Scene Parsing]()
  


## Peformance

 - Performance of Cityscapes validation set.

**Method**  | **Backbone** | **lr**     | **BatchSize**  | **epoch**    | **mean IoU (Single-scale)** |  **Trained weights**   |
------------|:------------:|:----------:|:--------------:|:------------:|:---------------------------:|------------------------|
PSPNet      | resnet101    |     0.01   |        8       | 80           | 78.1                        |  [pspnet_resnet_cityscapes_epoch_80.pdparams](https://pan.baidu.com/s/1adfvtq2JnLKRv_j7lOmW1A)|
GloRe      | resnet101    |     0.01   |        8       | 80           |  78.4                        |  [pspnet_resnet_pascalcontext_epoch_80.pdparams](https://pan.baidu.com/s/1r4SbrYKbVk38c0dXZLAi9w)              |


 - Performance of Pascal-context validation set.

**Method**  | **Backbone** | **lr**     | **BatchSize**  | **epoch**    | **mean IoU (Single-scale)** |  **Trained weights**   |
------------|:------------:|:----------:|:--------------:|:------------:|:---------------------------:|:----------------------:|
PSPNet       | resnet101    | 0.005       |   16            | 80           |   48.9                   |  [glore_resnet_cityscapes_epoch_80.pdparams](https://pan.baidu.com/s/1l7-sqt2DsUunD9l4YivgQw)                       |
GloRe       | resnet101    | 0.005       |   16            | 80           |    48.4                   |  [glore_resnet_pascalcontext_epoch_80.pdparams](https://pan.baidu.com/s/1rVuk7OfSj-AXR3ZCFGNmKg)                |


## Environment

This repo is developed under the following configurations:

 - Hardware: 4 GPUs for training, 1 GPU for testing
 - Software: Centos 6.10, ***CUDA>=9.2 Python>=3.6, Paddle>=1.6***


## Quick start: training and testing models

### 1. Preparing data

Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset. It should have this basic structure:

      cityscapes/
      ├── cityscapes_list
      │   ├── test.lst
      │   ├── train.lst
      │   ├── train+.lst
      │   ├── train++.lst
      │   ├── trainval.lst
      │   └── val.lst
      ├── gtFine
      │   ├── test
      │   ├── train
      │   └── val
      ├── leftImg8bit
      │   ├── test
      │   ├── train
      │   └── val
      ├── license.txt
      └── README
   
 Download Pascal-Context dataset. It should have this basic structure:  

      pascalContext/
      ├── GroundTruth_trainval_mat
      ├── GroundTruth_trainval_png
      ├── JPEGImages
      ├── pascal_context_train.txt
      ├── pascal_context_val.txt
      ├── README.md
      └── VOCdevkit

 Then, create symlinks for the Cityscapes and Pascal-Context datasets
 ```
 cd SemSegPaddle/data
 ln -s $cityscapes ./
 ln -s $pascalContext ./
 ```
 
### 2. Download pretrained weights
  Downlaod pretrained [resnet-101](https://pan.baidu.com/s/1niXBDZnLlUIulB7FY068DQ) weights file, and put it into the directory: ***./pretrained_model***
  
  Then, run the following command:
```
  tar -zxvf  ./repretrained/resnet101_v2.tgz -C pretrained_model 
```

### 3. Training

select confiure file for training according to the DECODER\_NAME, BACKBONE\_NAME and DATASET\_NAME.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch train.py  --use_gpu --use_mpio \
                                  --cfg ./configs/pspnet_res101_cityscapes.yaml 
```

### 4. Testing 
select confiure file for testing according to the DECODER\_NAME, BACKBONE\_NAME and DATASET\_NAME.

Single-scale testing:
```
CUDA_VISIBLE_DEVICES=0 python  eval.py --use_gpu \
                                       --use_mpio \
                                       --cfg ./configs/pspnet_res101_cityscapes.yaml 
```

Multi-scale testing:
```
CUDA_VISIBLE_DEVICES=0 python  eval.py --use_gpu \
                                       --use_mpio \
                                       --multi_scales \
                                       --cfg ./configs/pspnet_res101_cityscapes.yaml 
```

## Contact
If you have any questions regarding the repo, please create an issue.
