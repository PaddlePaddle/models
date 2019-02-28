# Paddle视频模型库

---
## 内容

- [安装](#安装)
- [简介](#简介)
- [数据准备](#数据准备)
- [模型库使用](#模型库使用)
- [模型简介](#模型简介)

## 安装

在当前模型库运行样例代码需要PadddlePaddle Fluid的v.1.2.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/install/index_cn.html)中的说明来更新PaddlePaddle。

## 简介
本次发布的是Paddle视频模型库第一期，包括五个视频分类模型。后续我们将会扩展到视频理解方向的更多应用场景以及视频编辑和生成等方向，以便为开发者提供简单、便捷的使用深度学习算法处理视频的途径。

Paddle视频模型库第一期主要包含如下模型。

| 模型 | 类别  | 描述 |
| :---------------: | :--------: | :------------:    |
| Attention Cluster | 视频分类| 百度自研模型，Kinetics600第一名最佳序列模型 |
| Attention LSTM | 视频分类| 常用模型，速度快精度高 |
| NeXtVLAD| 视频分类| 2nd-Youtube-8M最优单模型 |
| StNet| 视频分类| 百度自研模型，Kinetics600第一名模型之一 |
| TSN | 视频分类| 基于2D-CNN经典解决方案 |


## 数据准备

视频模型库使用Youtube-8M和Kinetics数据集, 具体使用方法请参考请参考[数据说明](./dataset/README.md)

## 模型库使用

视频模型库提供通用的train/test/infer框架，通过`train.py/test.py/infer.py`指定模型名、模型配置参数等可一键式进行训练和预测。

视频库目前支持的模型名如下：

1. AttentionCluster
2. AttentionLSTM
3. NEXTVLAD
4. STNET
5. TSN

Paddle提供默认配置文件位于`./configs`文件夹下，五种模型对应配置文件如下：

1. attention\_cluster.txt
2. attention\_lstm.txt
3. nextvlad.txt
4. stnet.txt
5. tsn.txt

### 模型训练

**预训练模型下载：** 视频模型库中StNet和TSN模型需要下载Resnet-50预训练模型，运行训练脚本会自动从[Resnet-50_pretrained](https://paddlemodels.bj.bcebos.com/video_classification/ResNet50_pretrained.tar.gz)下载预训练模型，存储于 ~/.paddle/weights/ 目录下，若该目录下已有已下载好的预训练模型，模型库会直接加载该预训练模型权重。

数据准备完毕后，可通过两种方式启动模型训练：

    python train.py --model-name=$MODEL_NAME --config=$CONFIG
            --save-dir=checkpoints --epoch=10 --log-interval=10 --valid-interval=1

    bash scripts/train/train_${MODEL_NAME}.sh

- 通过设置export CUDA\_VISIBLE\_DEVICES=0,1,2,3,4,5,6,7指定GPU卡训练。
- 可选参数见：
  
    ```
    python train.py --help
    ```

- 指定预训练模型可通过如下命令实现：

    ```
    python train.py --model-name=<$MODEL_NAME> --config=<$CONFIG>
            --pretrain=$PATH_TO_PRETRAIN
    ```

- 恢复训练模型可通过如下命令实现：

    ```
    python train.py --model-name=<$MODEL_NAME> --config=<$CONFIG>
            --resume=$PATH_TO_RESUME_WEIGHTS
    ```

### 模型评估

数据准备完毕后，可通过两种方式启动模型评估：

    python test.py --model-name=$MODEL_NAME --config=$CONFIG
            --log-interval=1 --weights=$PATH_TO_WEIGHTS

    bash scripts/test/test_${MODEL_NAME}.sh
  
- 通过设置export CUDA\_VISIBLE\_DEVICES=0使用GPU单卡评估。
- 可选参数见：
  
    ```
    python test.py --help
    ```

- 若模型评估未指定`--weights`参数，模型库会自动从[PaddleModels](https://paddlemodels.bj.bcebos.com)下载各模型已训练的Paddle release权重并完成模型评估，权重存储于`~/.paddle/weights/`目录下，若该目录下已有已下载好的预训练模型，模型库会直接加载该模型权重。

模型库各模型评估精度如下：


| 模型 | 数据集 | 精度类别  | 精度 |
| :---------------: | :-----------: | :-------: | :------: |
| AttentionCluster | Youtube-8M | GAP | 0.84 |
| AttentionLSTM | Youtube-8M | GAP | 0.86 |
| NeXtVLAD | Youtube=8M | GAP | 0.87 |
| stNet | Kinetics | Hit@1 | 0.69 |
| TSN | Kinetics | Hit@1 | 0.66 |

### 模型推断

模型推断可以通过各模型预测指定filelist中视频文件的类别，通过`infer.py`进行推断，可通过如下命令运行：

    python infer.py --model-name=$MODEL_NAME --config=$CONFIG
            --log-interval=1 --weights=$PATH_TO_WEIGHTS --filelist=$FILELIST

模型推断结果存储于`${MODEL_NAME}_infer_result`中，通过`pickle`格式存储。

- 通过设置export CUDA\_VISIBLE\_DEVICES=0使用GPU单卡推断。
- 可选参数见：
  
    ```
    python infer.py --help
    ```

- 若模型推断未使用`--weights`参数，模型库会自动下载Paddle release训练权重，参考[模型评估](#模型评估)

- 若模型推断未使用`--filelist`参数，则使用指定配置文件中配置的`filelist`。


## 模型简介

模型库各模型简介请参考：

1. [AttentionCluster](./models/attention_cluster/README.md)
2. [AttentionLSTM](./models/attention_lstm/README.md)
3. [NeXtVLAD](./models/nextvlad/README.md)
4. [StNet](./models/stnet/README.md)
5. [TSN](./models/tsn/README.md)

