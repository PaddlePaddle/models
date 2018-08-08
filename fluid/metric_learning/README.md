# Deep Metric Learning
Metric learning is a kind of methods to learn discriminative features for each sample, with the purpose that intra-class samples have smaller distances while inter-class samples have larger distances. With the develop of deep learning technique, metric learning methods are combined with deep neural networks to boost the performance of traditional tasks, such as face recognition/verification, human re-identification, image retrieval and so on. In this page, we introduce the way to implement deep metric learning using PaddlePaddle Fluid, including [data preparation](#data-preparation), [training](#training-a-model), [finetuning](#finetuning), [evaluation](#evaluation) and [inference](#inference).

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

An example for ImageNet classification is as follows. First of all, preparation of imagenet data can be done as:
```
cd data/ILSVRC2012/
sh download_imagenet2012.sh
```
The script ```data/split.py``` is used to split train/valid set. In our settings, we use images from first 100 classes(001-100) as training data while the other 100 classes are validation data. After the splitting, there are two label files which contain train and validation image labels respectively:

* *CUB200_train.txt*: label file of CUB-200 training set, with each line seperated by ```SPACE```, like:
```
current_path/images/097.Orchard_Oriole/Orchard_Oriole_0021_2432168643.jpg 97
current_path/images/097.Orchard_Oriole/Orchard_Oriole_0022_549995638.jpg 97
current_path/images/097.Orchard_Oriole/Orchard_Oriole_0034_2244771004.jpg 97
current_path/images/097.Orchard_Oriole/Orchard_Oriole_0010_2501839798.jpg 97
current_path/images/097.Orchard_Oriole/Orchard_Oriole_0008_491860362.jpg 97
current_path/images/097.Orchard_Oriole/Orchard_Oriole_0015_2545116359.jpg 97
...
```
* *CUB200_val.txt*: label file of CUB-200 validation set, with each line seperated by ```SPACE```, like.
```
current_path/images/154.Red_eyed_Vireo/Red_eyed_Vireo_0029_59210443.jpg 154
current_path/images/154.Red_eyed_Vireo/Red_eyed_Vireo_0021_2693953672.jpg 154
current_path/images/154.Red_eyed_Vireo/Red_eyed_Vireo_0016_2917350638.jpg 154
current_path/images/154.Red_eyed_Vireo/Red_eyed_Vireo_0027_2503540454.jpg 154
current_path/images/154.Red_eyed_Vireo/Red_eyed_Vireo_0026_2502710393.jpg 154
current_path/images/154.Red_eyed_Vireo/Red_eyed_Vireo_0022_2693134681.jpg 154
...
```

## Training metric learning models

To train a metric learning model, one need to set the neural network as backbone and the metric loss function to optimize. One example of training triplet loss using ResNet-50 is shown below:

```
python train.py  \
        --model=ResNet50 \
        --pretrained_model=${path_to_pretrain_model} \
        --lr=0.001 \
        --num_epochs=120 \
        --use_gpu=True \
        --train_batch_size=20 \
        --test_batch_size=20 \
        --loss_name=tripletloss \
        --model_save_dir="output_tripletloss"
```
**parameter introduction:**
* **model**: name model to use. Default: "SE_ResNeXt50_32x4d".
* **num_epochs**: the number of epochs. Default: 120.
* **batch_size**: the size of each mini-batch. Default: 256.
* **use_gpu**: whether to use GPU or not. Default: True.
* **model_save_dir**: the directory to save trained model. Default: "output".
* **lr**: initialized learning rate. Default: 0.1.
* **pretrained_model**: model path for pretraining. Default: None.

**training log:** the log from training ResNet-50 based triplet loss is like:
```
Pass 0, trainbatch 0, lr 9.99999974738e-05, loss_metric 0.0700866878033, loss_cls 5.23635625839, acc1 0.0, acc5 0.100000008941, time 0.16 sec
Pass 0, trainbatch 10, lr 9.99999974738e-05, loss_metric 0.0752244070172, loss_cls 5.30303478241, acc1 0.0, acc5 0.100000008941, time 0.14 sec
Pass 0, trainbatch 20, lr 9.99999974738e-05, loss_metric 0.0840565115213, loss_cls 5.41880941391, acc1 0.0, acc5 0.0333333350718, time 0.14 sec
Pass 0, trainbatch 30, lr 9.99999974738e-05, loss_metric 0.0698839947581, loss_cls 5.35385560989, acc1 0.0, acc5 0.0333333350718, time 0.14 sec
Pass 0, trainbatch 40, lr 9.99999974738e-05, loss_metric 0.0596057735384, loss_cls 5.34744024277, acc1 0.0, acc5 0.0, time 0.14 sec
Pass 0, trainbatch 50, lr 9.99999974738e-05, loss_metric 0.067836754024, loss_cls 5.37124729156, acc1 0.0, acc5 0.0333333350718, time 0.14 sec
Pass 0, trainbatch 60, lr 9.99999974738e-05, loss_metric 0.0637686774135, loss_cls 5.47412204742, acc1 0.0, acc5 0.0333333350718, time 0.14 sec
Pass 0, trainbatch 70, lr 9.99999974738e-05, loss_metric 0.0772982165217, loss_cls 5.38295936584, acc1 0.0, acc5 0.0, time 0.14 sec
Pass 0, trainbatch 80, lr 9.99999974738e-05, loss_metric 0.0861896127462, loss_cls 5.41250753403, acc1 0.0, acc5 0.0, time 0.14 sec
Pass 0, trainbatch 90, lr 9.99999974738e-05, loss_metric 0.0653102770448, loss_cls 5.53133153915, acc1 0.0, acc5 0.0, time 0.14 sec
...
```

## Finetuning

Finetuning is to finetune model weights in a specific task by loading pretrained weights. After initializing ```path_to_pretrain_model```, one can finetune a model as:
```
python train.py
       --model=SE_ResNeXt50_32x4d \
       --pretrained_model=${path_to_pretrain_model} \
       --batch_size=32 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --model_save_dir=output/ \
       --with_mem_opt=True \
       --lr_strategy=piecewise_decay \
       --lr=0.1
```

## Evaluation
Evaluation is to evaluate the performance of a trained model. One can download [pretrained models](#supported-models) and set its path to ```path_to_pretrain_model```. Then top1/top5 accuracy can be obtained by running the following command:
```
python eval.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=True \
       --pretrained_model=${path_to_pretrain_model}
```

According to the congfiguration of evaluation, the output log is like:
```
Testbatch 0,loss 2.1786134243, acc1 0.625,acc5 0.8125,time 0.48 sec
Testbatch 10,loss 0.898496925831, acc1 0.75,acc5 0.9375,time 0.51 sec
Testbatch 20,loss 1.32524681091, acc1 0.6875,acc5 0.9375,time 0.37 sec
Testbatch 30,loss 1.46830511093, acc1 0.5,acc5 0.9375,time 0.51 sec
Testbatch 40,loss 1.12802267075, acc1 0.625,acc5 0.9375,time 0.35 sec
Testbatch 50,loss 0.881597697735, acc1 0.8125,acc5 1.0,time 0.32 sec
Testbatch 60,loss 0.300163716078, acc1 0.875,acc5 1.0,time 0.48 sec
Testbatch 70,loss 0.692037761211, acc1 0.875,acc5 1.0,time 0.35 sec
Testbatch 80,loss 0.0969972759485, acc1 1.0,acc5 1.0,time 0.41 sec
...
```

## Inference
Inference is used to get prediction score or image features based on trained models.
```
python infer.py \
       --model=SE_ResNeXt50_32x4d \
       --batch_size=32 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=True \
       --pretrained_model=${path_to_pretrain_model}
```
The output contains predication results, including maximum score (before softmax) and corresponding predicted label.
```
Test-0-score: [13.168352], class [491]
Test-1-score: [7.913302], class [975]
Test-2-score: [16.959702], class [21]
Test-3-score: [14.197695], class [383]
Test-4-score: [12.607652], class [878]
Test-5-score: [17.725458], class [15]
Test-6-score: [12.678599], class [118]
Test-7-score: [12.353498], class [505]
Test-8-score: [20.828007], class [747]
Test-9-score: [15.135801], class [315]
Test-10-score: [14.585114], class [920]
Test-11-score: [13.739927], class [679]
Test-12-score: [15.040644], class [386]
...
```

## Supported models and performances

Models are trained by starting with learning rate ```0.1``` and decaying it by ```0.1``` after each pre-defined epoches, if not special introduced. Available top-1/top-5 validation accuracy on ImageNet 2012 are listed in table. Pretrained models can be downloaded by clicking related model names.

|model | ResNet50 | SE-ResNeXt-50
|- | - | -:
|triplet loss | 57.36% | 51.62%
|eml loss | 58.84% | 52.94%  
|quadruplet loss | 62.67% | 56.40%
