# Faster RCNN Objective Detection

---
## Table of Contents

- [Installation](#installation)
- [Introduction](#introduction)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Finetuning](#finetuning)
- [Evaluation](#evaluation)
- [Inference and Visualization](#inference-and-visualization)
- [Appendix](#appendix)

## Installation

Running sample code in this directory requires PaddelPaddle Fluid v0.13.0 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in [installation document](http://www.paddlepaddle.org/documentation/docs/zh/0.15.0/beginners_guide/install/install_doc.html#paddlepaddle) and make an update.

## Introduction

[Faster Rcnn](https://arxiv.org/abs/1506.01497) is a typical two stage detector. The total framework of network can be divided into four parts, as shown below:
<p align="center">
<img src="image/Faster_RCNN.jpg" height=400 width=400 hspace='10'/> <br />
Faster RCNN model
</p>

1. Base conv layer。As a CNN objective dection, Faster RCNN extract feature maps using a basic convolutional network. The feature maps then can be shared by RPN and fc layers. This sampel uses [ResNet-50](https://arxiv.org/abs/1512.03385) as base conv layer.
2. Region Proposal Network (RPN)。RPN generates proposals for detection。This block generates anchors by a set of size and ratio and classifies anchors into fore-ground and back-ground by softmax. Then refine anchors to obtain more precise proposals using box regression.
3. RoI pooling。This layer takes feature maps and proposals as input. The proposals are mapped to feature maps and pooled to the same size. The output are sent to fc layers for classification and regression.
4. Detection layer。Using the output of roi pooling to compute the class and locatoin of each proposal in two fc layers.

## Data preparation

Train the model on [MS-COCO dataset](http://cocodataset.org/#download), download dataset as below:

    cd dataset/coco
    ./download.sh


## Training

After data preparation, one can start the training step by:

    python train.py \
       --max_size=1333 \
       --scales=800 \
       --batch_size=8 \
       --batch_size_per_im=512 \
       --class_dim=81 \
       --model_save_dir=output/ \
       --max_iter=180000 \
       --learning_rate=0.01 \
       --padding_minibatch=True

- Set ```export CUDA_VISIBLE_DEVICES=0,1``` to specifiy the id of GPU you want to use.
- For more help on arguments:

    python train.py --help

**data reader introduction:**

* Data reader is defined in `reader.py`.
* Scaling the short side of all images to `scales`. If the long side is larger than `max_size`, then scaling the long side to `max_size`.
* In training stage, images are horizontally flipped.
* Images in the same batch can be padding to the same size.

**model configuration:**

* Roi_pool layer takes average pooling.
* NMS threshold=0.7. During training, pre\_nms=12000, post\_nms=2000; during test, pre\_nms=6000, post\_nms=1000.
* In generating proposal lables, fg\_fraction=0.25, fg\_thresh=0.5, bg\_thresh_hi=0.5, bg\_thresh\_lo=0.0.
* In rpn target assignment, rpn\_fg\_fraction=0.5, rpn\_positive\_overlap=0.7, rpn\_negative\_overlap=0.3.

**training strategy:**

*  Use momentum optimizer with momentum=0.9.
*  Weight decay is 0.0001.
*  In first 500 iteration, the learning rate increases linearly from 0.00333 to 0.01. Then lr is decayed at 120000, 160000 iteration with multiplier 0.1, 0.01. The maximum iteration is 180000.
*  Set the learning rate of bias to two times as global lr in non basic convolutional layers.
*  In basic convolutional layers, parameters of affine layers and res body do not update.
*  Use Nvidia Tesla V100 8GPU, total time for training is about 40 hours.

Training result is shown as below：
<p align="center">
<img src="image/train_loss.jpg" height=500 width=650 hspace='10'/> <br />
Faster RCNN train loss
</p>

## Finetuning

Finetuning is to finetune model weights in a specific task by loading pretrained weights. After initializing ```pretrained_model```, one can finetune a model as:

    python train.py
        --max_size=1333 \
        --scales=800 \
        --pretrained_model=${path_to_pretrain_model} \
        --batch_size= 8\
        --model_save_dir=output/ \
        --class_dim=81 \
        --max_iter=180000 \
        --learning_rate=0.01

## Evaluation

Evaluation is to evaluate the performance of a trained model. This sample provides `eval_coco_map.py` which uses a COCO-specific mAP metric defined by [COCO committee](http://cocodataset.org/#detections-eval). To use `eval_coco_map.py` , [cocoapi](https://github.com/cocodataset/cocoapi) is needed. Install the cocoapi:

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

`eval_coco_map.py` is the main executor for evalution, one can start evalution step by:

    python eval_coco_map.py \
        --dataset=coco2017 \
        --pretrained_mode=${path_to_pretrain_model} \
        --batch_size=1 \
        --nms_threshold=0.5 \
        --score_threshold=0.05

Evalutaion result is shown as below:
<p align="center">
<img src="image/mAP.jpg" height=500 width=650 hspace='10'/> <br />
Faster RCNN mAP
</p>

## Inference and Visualization

Inference is used to get prediction score or image features based on trained models. `infer.py`  is the main executor for inference, one can start infer step by:

    python infer.py \
       --dataset=coco2017 \
        --pretrained_model=${path_to_pretrain_model}  \
        --image_path=data/COCO17/val2017/  \
        --image_name=000000000139.jpg \
        --draw_threshold=0.6

Visualization of infer result is shown as below:
<p align="center">
<img src="image/000000000139.jpg" height=300 width=400 hspace='10'/>
<img src="image/000000127517.jpg" height=300 width=400 hspace='10'/>
<img src="image/000000203864.jpg" height=300 width=400 hspace='10'/>
<img src="image/000000515077.jpg" height=300 width=400 hspace='10'/> <br />
Faster RCNN Visualization Examples
</p>

## Appendix

### Debug model

Because of the complicated model and many parameters, one needs to debug each block in model.

#### Input

Fluid does variable assignment as below:

    rpn_rois_t = np.load('rpn_rois')
    rpn_rois = fluid.core.LoDTensor()
    rpn_rois.set(rpn_rois_t, place)

If the input variable is LoDTensor, one needs to set lod:

    rpn_rois.set_lod(lod)

In addition, `use_random` should set to False in debug mode.

#### Output

In debug mode, one may want to find the name of variables. Two methods are provided as below:

1. use program.global_block().var to get all variables then find the name.
2. set variable.persistable to True and print the variable to get the name.

To get the output of variables, three methods are provided as below:

1. use get\_var and the name of variable to fetch the output in exe.run, for example:

        rpn_conv = fluid.get_var('conv_rpn.tmp_1')
        rpn_conv_v = exe.run(fetch_list=rpn_conv)

2. after exe.run, use global_scope().find_var to get the output of variable, for example:

        rpn_conv_v = fluid.global_scope().find_var('conv_rpn.tmp_1').get_tensor()

3. use Print op to print the output directly, for example:

        fluid.layers.Print(rpn_conv)
