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

Caltech-UCSD Birds 200 (CUB-200) is an image dataset including 200 bird species. We use it to conduct the metric learning experiments. More details of this dataset can be found from its [official website](http://www.vision.caltech.edu/visipedia/CUB-200.html). First of all, preparation of CUB-200 data can be done as:
```
cd data/
sh download_cub200.sh
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

## Evaluation
Evaluation is to evaluate the performance of a trained model. One can download [pretrained models](#supported-models) and set its path to ```path_to_pretrain_model```. Then Recall@Rank-1 can be obtained by running the following command:
```
python eval.py \
       --model=ResNet50 \
       --pretrained_model=${path_to_pretrain_model} \
       --batch_size=30 \
       --loss_name=tripletloss
```

According to the congfiguration of evaluation, the output log is like:
```
testbatch 0, loss 17.0384693146, recall 0.133333333333, time 0.08 sec
testbatch 10, loss 15.4248628616, recall 0.2, time 0.07 sec
testbatch 20, loss 19.3986873627, recall 0.0666666666667, time 0.07 sec
testbatch 30, loss 19.8149013519, recall 0.166666666667, time 0.07 sec
testbatch 40, loss 18.7500724792, recall 0.0333333333333, time 0.07 sec
testbatch 50, loss 15.1477527618, recall 0.166666666667, time 0.07 sec
testbatch 60, loss 21.6039619446, recall 0.0666666666667, time 0.07 sec
testbatch 70, loss 16.3203811646, recall 0.1, time 0.08 sec
testbatch 80, loss 17.3300457001, recall 0.133333333333, time 0.14 sec
testbatch 90, loss 17.9943237305, recall 0.0333333333333, time 0.07 sec
testbatch 100, loss 20.4538421631, recall 0.1, time 0.07 sec
End test, test_loss 18.2126255035, test recall 0.573597359736
...
```

## Inference
Inference is used to get prediction score or image features based on trained models.
```
python infer.py --model=ResNet50 \
                --pretrained_model=${path_to_pretrain_model}
```
The output contains learned feature for each test sample:
```
Test-0-feature: [0.1551965  0.48882252 0.3528545  ... 0.35809007 0.6210782 0.34474897]
Test-1-feature: [0.26215672 0.71406883 0.36118034 ... 0.4711366  0.6783772 0.26591945]
Test-2-feature: [0.26164916 0.46013424 0.38381338 ... 0.47984493 0.5830286 0.22124235]
Test-3-feature: [0.22502825 0.44153655 0.29287377 ... 0.45510024 0.81386226 0.21451607]
Test-4-feature: [0.27748746 0.49068335 0.28269237 ... 0.47356504 0.73254013 0.22317657]
Test-5-feature: [0.17743547 0.5232162  0.35012805 ... 0.38921246 0.80238944 0.26693743]
Test-6-feature: [0.18314484 0.4294481  0.37652573 ... 0.4795592  0.7446839 0.24178651]
Test-7-feature: [0.25836483 0.49866533 0.3469289  ... 0.38316026 0.56015515 0.22388287]
Test-8-feature: [0.30613047 0.5200348  0.2847372  ... 0.5700768  0.76645917 0.26504722]
Test-9-feature: [0.3305695  0.46257797 0.27108437 ... 0.42891273 0.5112956 0.26442713]
Test-10-feature: [0.16024818 0.46871603 0.32608703 ... 0.3341719  0.6876993 0.26097256]
Test-11-feature: [0.37611157 0.6006333  0.3023942  ... 0.4729057  0.53841203 0.19621202]
Test-12-feature: [0.17515017 0.41597834 0.45567667 ... 0.45650777 0.5987687 0.25734115]
...
```

## Performances

For comparation, many metric learning models with different neural networks and loss functions are trained using corresponding experiential parameters. Recall@Rank-1 is used as evaluation metric and the performance is listed in the table. Pretrained models can be downloaded by clicking related model names.

|model | ResNet50 | SE-ResNeXt-50
|- | - | -:
|[triplet loss]() | 57.36% | 51.62%
|[eml loss]() | 58.84% | 52.94%  
|[quadruplet loss]() | 62.67% | 56.40%
