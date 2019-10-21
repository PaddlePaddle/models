# Deep Metric Learning
Metric learning is a kind of methods to learn discriminative features for each sample, with the purpose that intra-class samples have smaller distances while inter-class samples have larger distances in the learned space. With the develop of deep learning technique, metric learning methods are combined with deep neural networks to boost the performance of traditional tasks, such as face recognition/verification, human re-identification, image retrieval and so on. In this page, we introduce the way to implement deep metric learning using PaddlePaddle Fluid, including [data preparation](#data-preparation), [training](#training-metric-learning-models), [finetuning](#finetuning), [evaluation](#evaluation), [inference](#inference) and [Performances](#performances).

---
## Table of Contents
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training metric learning models](#training-metric-learning-models)
- [Finetuning](#finetuning)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Performances](#performances)

## Installation

Running sample code in this directory requires PaddelPaddle Fluid v0.14.0 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in [installation document](http://paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/index_cn.html) and make an update.

## Data preparation

Stanford Online Product(SOP) dataset contains 120,053 images of 22,634 products downloaded from eBay.com. We use it to conduct the metric learning experiments. For training, 59,551 out of 11,318 classes are used, and 11,316 classes(60,502 images) are held out for testing. First of all, preparation of SOP data can be done as:
```
cd data/
sh download_sop.sh
```

SOP dataset directory structure like this:
```
data/
├──Stanford_Online_Products
│   ├── bicycle_final
│   ├── bicycle_final.txt
│   ├── cabinet_final
│   ├── cabinet_final.txt
|   ...
├──Stanford_Online_Products.zip 
```

if you are Windows user, you can download the dataset using below ftp link and unzip file to ```data``` directory.
```
SOP: ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
```

## Training metric learning models

To train a metric learning model, one need to set the neural network as backbone and the metric loss function to optimize. You can download [ResNet50](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar) pretrained on imagenet dataset as backbone. We train meiric learning model using softmax or arcmargin loss firstly, and then fine-turned the model using other metric learning loss, such as triplet, quadruplet and eml loss. One example of training using arcmargin loss is shown below:


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
* **loss_name**: loss for training model. Default: "softmax".
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
Evaluation is to evaluate the performance of a trained model. You should set model path to ```path_to_pretrain_model```. Then Recall@Rank-1 can be obtained by running the following command:
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

For comparation, many metric learning models with different neural networks and loss functions are trained using corresponding experiential parameters. Recall@Rank-1 is used as evaluation metric and the performance is listed in the table.

|pretrain model | softmax | arcmargin
|- | - | -:
|without fine-tuned | 77.42% | 78.11%
|fine-tuned with triplet | 78.37% | 79.21%
|fine-tuned with quadruplet | 78.10% | 79.59%
|fine-tuned with eml | 79.32% | 80.11%
|fine-tuned with npairs | - | 79.81%

## Reference

- ArcFace: Additive Angular Margin Loss for Deep Face Recognition [link](https://arxiv.org/abs/1801.07698)
- Margin Sample Mining Loss: A Deep Learning Based Method for Person Re-identification [link](https://arxiv.org/abs/1710.00478)
- Large Scale Strongly Supervised Ensemble Metric Learning, with Applications to Face Verification and Retrieval [link](https://arxiv.org/abs/1212.6094)
- Improved Deep Metric Learning with Multi-class N-pair Loss Objective [link](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)
