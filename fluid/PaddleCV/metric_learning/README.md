# Deep Metric Learning
Metric learning is a kind of methods to learn discriminative features for each sample, with the purpose that intra-class samples have smaller distances while inter-class samples have larger distances in the learned space. With the develop of deep learning technique, metric learning methods are combined with deep neural networks to boost the performance of traditional tasks, such as face recognition/verification, human re-identification, image retrieval and so on. In this page, we introduce the way to implement deep metric learning using PaddlePaddle Fluid, including [data preparation](#data-preparation), [training](#training-a-model), [finetuning](#finetuning), [evaluation](#evaluation) and [inference](#inference).

---
## Table of Contents
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training metric learning models](#training-a-model)
- [Finetuning](#finetuning)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Performances](#supported-models)

## Installation

Running sample code in this directory requires PaddelPaddle Fluid v0.14.0 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in [installation document](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html) and make an update.

## Data preparation

Stanford Online Product(SOP) dataset contains 120,053 images of 22,634 products downloaded from eBay.com. We use it to conduct the metric learning experiments. For training, 59,5511 out of 11,318 classes are used, and 11,316 classes(60,502 images) are held out for testing. First of all, preparation of SOP data can be done as:
```
cd data/
sh download_sop.sh
```

## Training metric learning models

To train a metric learning model, one need to set the neural network as backbone and the metric loss function to optimize. We train meiric learning model using softmax or [arcmargin](https://arxiv.org/abs/1801.07698) loss firstly, and then fine-turned the model using other metric learning loss, such as triplet, [quadruplet](https://arxiv.org/abs/1710.00478) and [eml](https://arxiv.org/abs/1212.6094) loss. One example of training using arcmargin loss is shown below:


```
python train_elem.py  \
        --model=ResNet50 \
        --train_batch_size=256 \
        --test_batch_size=50 \
        --lr=0.01 \
        --total_iter_num=30000 \
        --use_gpu=True \
        --pretrained_model=${path_to_pretrain_imagenet_model} \
        --model_save_dir=${output_model_path} \
        --loss_name=arcmargin \
        --arc_scale=80.0 \ 
        --arc_margin=0.15 \
        --arc_easy_margin=False
```
**parameter introduction:**
* **model**: name model to use. Default: "ResNet50".
* **train_batch_size**: the size of each training mini-batch. Default: 256.
* **test_batch_size**: the size of each testing mini-batch. Default: 50.
* **lr**: initialized learning rate. Default: 0.01.
* **total_iter_num**: total number of training iterations. Default: 30000.
* **use_gpu**: whether to use GPU or not. Default: True.
* **pretrained_model**: model path for pretraining. Default: None.
* **model_save_dir**: the directory to save trained model. Default: "output".
* **loss_name**: loss fortraining model. Default: "softmax".
* **arc_scale**: parameter of arcmargin loss. Default: 80.0.
* **arc_margin**: parameter of arcmargin loss. Default: 0.15.
* **arc_easy_margin**: parameter of arcmargin loss. Default: False.

## Finetuning

Finetuning is to finetune model weights in a specific task by loading pretrained weights. After training model using softmax or arcmargin loss, one can finetune the model using triplet, quadruplet or eml loss. One example of fine-turned using eml loss is shown below:

```
python train_pair.py  \
        --model=ResNet50 \
        --train_batch_size=160 \
        --test_batch_size=50 \
        --lr=0.0001 \
        --total_iter_num=100000 \
        --use_gpu=True \
        --pretrained_model=${path_to_pretrain_arcmargin_model} \
        --model_save_dir=${output_model_path} \
        --loss_name=eml \
        --samples_each_class=2
```

## Evaluation
Evaluation is to evaluate the performance of a trained model. One can download [pretrained models](#supported-models) and set its path to ```path_to_pretrain_model```. Then Recall@Rank-1 can be obtained by running the following command:
```
python eval.py \
       --model=ResNet50 \
       --batch_size=50 \
       --pretrained_model=${path_to_pretrain_model} \
```

## Inference
Inference is used to get prediction score or image features based on trained models.
```
python infer.py \
       --model=ResNet50 \
       --batch_size=1 \         
       --pretrained_model=${path_to_pretrain_model}
```

## Performances

For comparation, many metric learning models with different neural networks and loss functions are trained using corresponding experiential parameters. Recall@Rank-1 is used as evaluation metric and the performance is listed in the table. Pretrained models can be downloaded by clicking related model names.

|pretrain model | softmax | arcmargin
|- | - | -:
|without fine-tuned | 77.42% | 78.11%
|fine-tuned with triplet | 78.37% | 79.21%
|fine-tuned with quadruplet | 78.10% | 79.59%
|fine-tuned with eml | 79.32% | 80.11%
