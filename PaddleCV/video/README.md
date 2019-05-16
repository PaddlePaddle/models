
## 简介
本教程期望给开发者提供基于PaddlePaddle的便捷、高效的使用深度学习算法解决视频理解、视频编辑、视频生成等一系列模型。目前包含视频分类模型，后续会不断的扩展到其他更多场景。

目前视频分类模型包括:

| 模型 | 类别  | 描述 |
| :--------------- | :--------: | :------------: |
| [Attention Cluster](./models/attention_cluster/README.md) | 视频分类| CVPR'18提出的视频多模态特征注意力聚簇融合方法 |
| [Attention LSTM](./models/attention_lstm/README.md)  | 视频分类| 常用模型，速度快精度高 |
| [NeXtVLAD](./models/nextvlad/README.md)  | 视频分类| 2nd-Youtube-8M最优单模型 |
| [StNet](./models/stnet/README.md)  | 视频分类| AAAI'19提出的视频联合时空建模方法 |
| [TSM](./models/tsm/README.md) | 视频分类| 基于时序移位的简单高效视频时空建模方法 |
| [TSN](./models/tsn/README.md) | 视频分类| ECCV'16提出的基于2D-CNN经典解决方案 |
| [Non-local](./models/nonlocal_model/README.md) | 视频分类| 视频非局部关联建模模型 |

### 主要特点

- 包含视频分类方向的多个主流领先模型，其中Attention LSTM，Attention Cluster和NeXtVLAD是比较流行的特征序列模型，Non-local, TSN, TSM和StNet是End-to-End的视频分类模型。Attention LSTM模型速度快精度高，NeXtVLAD是2nd-Youtube-8M比赛中最好的单模型, TSN是基于2D-CNN的经典解决方案，TSM是基于时序移位的简单高效视频时空建模方法，Non-local模型提出了视频非局部关联建模方法。Attention Cluster和StNet是百度自研模型，分别发表于CVPR2018和AAAI2019，是Kinetics600比赛第一名中使用到的模型。

- 提供了适合视频分类任务的通用骨架代码，用户可一键式高效配置模型完成训练和评测。

## 安装

在当前模型库运行样例代码需要PadddlePaddle Fluid v.1.4.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://www.paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html)中的说明来更新PaddlePaddle。

## 数据准备

视频模型库使用Youtube-8M和Kinetics数据集, 具体使用方法请参考[数据说明](./dataset/README.md)

## 快速使用

视频模型库提供通用的train/test/infer框架，通过`train.py/test.py/infer.py`指定模型名、模型配置参数等可一键式进行训练和预测。

以StNet模型为例：

单卡训练：

``` bash
export CUDA_VISIBLE_DEVICES=0
python train.py --model_name=STNET
        --config=./configs/stnet.txt
        --save_dir=checkpoints
```

多卡训练：

``` bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --model_name=STNET
        --config=./configs/stnet.txt
        --save_dir=checkpoints
```

视频模型库同时提供了快速训练脚本，脚本位于`scripts/train`目录下，可通过如下命令启动训练:

``` bash
bash scripts/train/train_stnet.sh
```

- 请根据`CUDA_VISIBLE_DEVICES`指定卡数修改`config`文件中的`num_gpus`和`batch_size`配置。

### 注意

使用Windows GPU环境的用户，需要将示例代码中的[fluid.ParallelExecutor](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/fluid_cn.html#parallelexecutor)替换为[fluid.Executor](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/fluid_cn.html#executor)。

## 模型库结构

### 代码结构

```
configs/
  stnet.txt
  tsn.txt
  ...
dataset/
  youtube/
  kinetics/
datareader/
  feature_readeer.py
  kinetics_reader.py
  ...
metrics/
  kinetics/
  youtube8m/
  ...
models/
  stnet/
  tsn/
  ...
scripts/
  train/
  test/
train.py
test.py
infer.py
```

- `configs`: 各模型配置文件模板
- `datareader`: 提供Youtube-8M，Kinetics数据集reader
- `metrics`: Youtube-8，Kinetics数据集评估脚本
- `models`: 各模型网络结构构建脚本
- `scripts`: 各模型快速训练评估脚本
- `train.py`: 一键式训练脚本，可通过指定模型名，配置文件等一键式启动训练
- `test.py`: 一键式评估脚本，可通过指定模型名，配置文件，模型权重等一键式启动评估
- `infer.py`: 一键式推断脚本，可通过指定模型名，配置文件，模型权重，待推断文件列表等一键式启动推断

## Model Zoo

- 基于Youtube-8M数据集模型：

| 模型 | Batch Size | 环境配置 | cuDNN版本 | GAP | 下载链接 |
| :-------: | :---: | :---------: | :-----: | :----: | :----------: |
| Attention Cluster | 2048 | 8卡P40 | 7.1 | 0.84 | [model](https://paddlemodels.bj.bcebos.com/video_classification/attention_cluster_youtube8m.tar.gz) |
| Attention LSTM | 1024 | 8卡P40 | 7.1 | 0.86 | [model](https://paddlemodels.bj.bcebos.com/video_classification/attention_lstm_youtube8m.tar.gz) |
| NeXtVLAD | 160 | 4卡P40 | 7.1 | 0.87 | [model](https://paddlemodels.bj.bcebos.com/video_classification/nextvlad_youtube8m.tar.gz) |

- 基于Kinetics数据集模型：

| 模型 | Batch Size | 环境配置 | cuDNN版本 | Top-1 | 下载链接 |
| :-------: | :---: | :---------: | :----: | :----: | :----------: |
| StNet | 128 | 8卡P40 | 7.1 | 0.69 | [model](https://paddlemodels.bj.bcebos.com/video_classification/stnet_kinetics.tar.gz) |
| TSN | 256 | 8卡P40 | 7.1 | 0.67 | [model](https://paddlemodels.bj.bcebos.com/video_classification/tsn_kinetics.tar.gz) |
| TSM | 128 | 8卡P40 | 7.1 | 0.70 | [model](https://paddlemodels.bj.bcebos.com/video_classification/tsm_kinetics.tar.gz) |
| Non-local | 64 | 8卡P40 | 7.1 | 0.74 | [model](https://paddlemodels.bj.bcebos.com/video_classification/nonlocal_kinetics.tar.gz) |
## 参考文献

- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
- [Beyond Short Snippets: Deep Networks for Video Classification](https://arxiv.org/abs/1503.08909) Joe Yue-Hei Ng, Matthew Hausknecht, Sudheendra Vijayanarasimhan, Oriol Vinyals, Rajat Monga, George Toderici
- [NeXtVLAD: An Efficient Neural Network to Aggregate Frame-level Features for Large-scale Video Classification](https://arxiv.org/abs/1811.05014), Rongcheng Lin, Jing Xiao, Jianping Fan
- [StNet:Local and Global Spatial-Temporal Modeling for Human Action Recognition](https://arxiv.org/abs/1811.01549), Dongliang He, Zhichao Zhou, Chuang Gan, Fu Li, Xiao Liu, Yandong Li, Limin Wang, Shilei Wen
- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool
- [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383v1), Ji Lin, Chuang Gan, Song Han
- [Non-local Neural Networks](https://arxiv.org/abs/1711.07971v1), Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He
## 版本更新

- 3/2019: 新增模型库，发布Attention Cluster，Attention LSTM，NeXtVLAD，StNet，TSN五个视频分类模型。
- 4/2019: 发布Non-local, TSM两个视频分类模型。
