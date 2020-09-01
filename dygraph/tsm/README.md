# TSM 视频分类模型

本目录下为基于PaddlePaddle 动态图实现的 TSM视频分类模型，静态图实现请参考[TSM 视频分类模型](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo/models/tsm)

---
## 内容

- [模型简介](#模型简介)
- [安装说明](#安装说明)
- [数据准备](#数据准备)
- [模型训练](#模型训练)


## 模型简介

Temporal Shift Module是由MIT和IBM Watson AI Lab的Ji Lin，Chuang Gan和Song Han等人提出的通过时间位移来提高网络视频理解能力的模块, 详细内容请参考论文[Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383v1)

## 安装说明

1. 在当前模型库运行样例代码需要PaddlePaddle v.2.0.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://www.paddlepaddle.org/documentation/docs/zh/1.6/beginners_guide/install/index_cn.html)中的说明来更新PaddlePaddle。
2. 下载模型repo: git clone https://github.com/PaddlePaddle/models 

### 其他环境依赖

- Python >= 3.7

- CUDA >= 8.0

- CUDNN >= 7.0


## 数据准备

TSM的训练数据采用UCF101行为识别数据集,包含101个行为类别。
ucf101_reader.py文件中的ucf101_root设置为ucf101数据集目录，其中的videos、rawframes分别为视频格式和帧图格式，大小分别为6.8G、56G。
准备数据步骤：
1. 下载官方ucf101数据: wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar, 解压存放到$ucf101_root/videos
2. 提取视频frames文件(TODO),存放到$ucf101_root/frames
3. 生成video文件路径list文件(步骤TODO),存放到./data/dataset/ucf101/


## 模型训练

数据准备完毕后，可以通过如下方式启动训练.  

- 从头开始训练
sh run_ucf101.sh

- 基于imagenet pretrain的resnet backbone参数进行训练:

1. 需要加载在ImageNet上训练的ResNet50权重作为初始化参数，wget https://paddlemodels.bj.bcebos.com/video_classification/ResNet50_pretrained.tar.gz, 并解压
2. 通过--weights=./ResNet50_pretrained/启动训练: sh run_ucf101_imagenet.sh

- 基于k400 pretrain模型进行finetune:

1. 下载静态图已发布模型 wget https://paddlemodels.bj.bcebos.com/video_classification/TSM.pdparams 
2. mkdir k400_wei &&  mv TSM.pdparams k400_wei
3. 通过--weights=k400_wei/TSM.pdparams启动训练: sh run_ucf101_k400.sh

在UCF101数据集下：

|Top-1|Top-5|pretrain|
|:-:|:-:|:-:|
|84.37%|95.68%|ImageNet|
|94.54%|98.96%|Kinetics-400|

