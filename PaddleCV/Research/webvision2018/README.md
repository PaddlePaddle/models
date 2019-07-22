# WebVision Image Classification 2018 Challenge 
The goal of this challenge is to advance the area of learning knowledge and representation from web data. The web data not only contains huge numbers of visual images, but also rich meta information concerning these visual data, which could be exploited to learn good representations and models. 
More detail [[WebVision2018](https://www.vision.ee.ethz.ch/webvision/challenge.html)].

By observing the web data, we find that there are five key challenges, i.e., imbalanced class sizes, high intra-classes diversity and inter-class similarity, imprecise instances,
insufficient representative instances, and ambiguous class labels. To alleviate these challenges, we assume that every training instance has
the potential to contribute positively by alleviating the data bias and noise via reweighting the influence of each instance according to different
class sizes, large instance clusters, its confidence, small instance bags and the labels. In this manner, the influence of bias and noise in the
web data can be gradually alleviated, leading to the steadily improving performance of URNet. Experimental results in the WebVision 2018
challenge with 16 million noisy training images from 5000 classes show that our approach outperforms state-of-the-art models and ranks the first
place in the image classification task. The detail of our solution can refer to our paper[[URNet](https://arxiv.org/abs/1811.00700)].

## 1.Prepare data
We have provided a download + preprocess script of valset data.
```
cd data
sh download_webvision2018.sh
```
Note that the server hosting Webvision Data reboots every day at midnight (Zurich time). You might want to change wget to something else. 

## 2.Environment installation
Cudnn >= 7, CUDA 8/9, PaddlePaddle version >= 1.3, python version 2.7 （More detail [[PaddlePaddle](https://github.com/paddlepaddle/paddle)]）

## 3.Download pretrained model
| Model | Acc@1 | Acc@5
| - | - | -
| [ResNeXt101_32x4d](https://paddlemodels.bj.bcebos.com/webvision/ResNeXt101_32x4d_Released.tar.gz) | 53.4% | 77.1%

## 4.Test image 
```
sh run.sh
```
or
```
export CUDA_VISIBLE_DEVICES=$GPU_ID
export FLAGS_fraction_of_gpu_memory_to_use=1.0
python infer.py --model ResNeXt101_32x4d \
                --pretrained_model $PRETRAINEDMODELPATH \
                --class_dim 5000 \
                --img_path $IMGPATH \
                --img_list $IMGLIST \
                --use_gpu True
```

You will get the predictions of images.
## 5.Evaluation
```
export CUDA_VISIBLE_DEVICES=$GPU_ID
export FLAGS_fraction_of_gpu_memory_to_use=1.0
python eval.py  --model ResNeXt101_32x4d \
                --pretrained_model $PRETRAINEDMODELPATH \
                --class_dim 5000 \
                --img_path $IMGPATH \
                --img_list $IMGLIST \
                --use_gpu True

```
You will get the Acc@1 and Acc@5.

