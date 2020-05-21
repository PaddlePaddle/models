
## 简介
本教程期望给开发者提供基于PaddlePaddle的便捷、高效的使用深度学习算法解决视频理解、视频编辑、视频生成等一系列模型。目前包含视频分类和动作定位模型，后续会不断的扩展到其他更多场景。

目前视频分类和动作定位模型包括:

| 模型 | 类别  | 描述 |
| :--------------- | :--------: | :------------: |
| [Attention Cluster](./models/attention_cluster/README.md) | 视频分类| CVPR'18提出的视频多模态特征注意力聚簇融合方法 |
| [Attention LSTM](./models/attention_lstm/README.md)  | 视频分类| 常用模型，速度快精度高 |
| [NeXtVLAD](./models/nextvlad/README.md)  | 视频分类| 2nd-Youtube-8M比赛第3名模型 |
| [StNet](./models/stnet/README.md)  | 视频分类| AAAI'19提出的视频联合时空建模方法 |
| [TSM](./models/tsm/README.md) | 视频分类| 基于时序移位的简单高效视频时空建模方法 |
| [TSN](./models/tsn/README.md) | 视频分类| ECCV'16提出的基于2D-CNN经典解决方案 |
| [Non-local](./models/nonlocal_model/README.md) | 视频分类| 视频非局部关联建模模型 |
| [C-TCN](./models/ctcn/README.md) | 视频动作定位| 2018年ActivityNet夺冠方案 |
| [BSN](./models/bsn/README.md) | 视频动作定位| 为视频动作定位问题提供高效的proposal生成方法 |
| [BMN](./models/bmn/README.md) | 视频动作定位| 2019年ActivityNet夺冠方案 |
| [ETS](./models/ets/README.md) | 视频描述| ICCV'15提出的结合时序注意力机制的建模方法 |
| [TALL](./models/tall/README.md) | 视频查找| ICCV'17多模态时序回归定位方法 |

### 主要特点

- 包含视频分类和动作定位方向的多个主流领先模型，其中Attention LSTM，Attention Cluster和NeXtVLAD是比较流行的特征序列模型，Non-local, TSN, TSM和StNet是End-to-End的视频分类模型。Attention LSTM模型速度快精度高，NeXtVLAD是2nd-Youtube-8M比赛中最好的单模型, TSN是基于2D-CNN的经典解决方案，TSM是基于时序移位的简单高效视频时空建模方法，Non-local模型提出了视频非局部关联建模方法。Attention Cluster和StNet是百度自研模型，分别发表于CVPR2018和AAAI2019，是Kinetics600比赛第一名中使用到的模型。C-TCN动作定位模型也是百度自研，2018年ActivityNet比赛的夺冠方案。BSN模型采用自底向上的方法生成proposal，为视频动作定位问题中proposal的生成提供高效的解决方案。BMN模型是百度自研模型，2019年ActivityNet夺冠方案。ETS结合时序注意力机制构建网络，是视频生成文字描述的经典模型。TALL是利用多模态时序回归定位器对视频片段进行查找的模型。

- 提供了适合视频分类和动作定位任务的通用骨架代码，用户可一键式高效配置模型完成训练和评测。

### 推荐用法

- 视频分类共开源7个模型，可分为：端到端模型、序列模型。端到端模型：TSN推荐在时序不敏感视频场景（比如互联网视频场景）使用；TSM、StNet推荐在时序敏感视频场景（比如Kinetics数据集）使用；Non-local模型计算量较大，在科研场景推荐。序列模型：Attention LSTM，Attention Cluster和NeXtVLAD 整体性能接近，但是网络结构不同，推荐集成多个模型使用。

- 视频动作定位共开源3个模型，视频动作定位推荐使用CTCN模型，时序提名生成推荐使用BMN模型。


## 安装

在当前模型库运行样例代码需要PaddlePaddle Fluid v.1.6.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://www.paddlepaddle.org/documentation/docs/zh/1.6/beginners_guide/install/index_cn.html)中的说明来更新PaddlePaddle。

### 其他环境依赖

- Python >= 2.7

- CUDA >= 8.0

- CUDNN >= 7.0

- pandas

- h5py

- 使用Youtube-8M数据集时，需要将tfrecord数据转化成pickle格式，需要用到Tensorflow，详见[数据说明](./data/dataset/README.md)中Youtube-8M部分。与此相关的模型是Attention Cluster, Attention LSTM, NeXtVLAD，使用其他模型请忽略此项。

- 使用Kinetics数据集时，如果需要将mp4文件提前解码并保存成pickle格式，需要用到ffmpeg，详见[数据说明](./data/dataset/README.md)中Kinetics部分。需要说明的是Nonlocal模型虽然也使用Kinetics数据集，但输入数据是视频源文件，不需要提前解码，不涉及此项。与此相关的模型是TSN, TSM, StNet，使用其他模型请忽略此项。

## 数据准备

视频模型库使用Youtube-8M和Kinetics数据集, 具体使用方法请参考[数据说明](./data/dataset/README.md)

## 快速使用

视频模型库提供通用的train/evaluate/predict框架，通过`train.py/eval.py/predict.py`指定任务类型、模型名、模型配置参数等可一键式进行训练和预测。

以StNet模型为例：

单卡训练：

``` bash
export CUDA_VISIBLE_DEVICES=0
python train.py --model_name=STNET \
                --config=./configs/stnet.yaml \
                --log_interval=10 \
                --valid_interval=1 \
                --use_gpu=True \
                --save_dir=./data/checkpoints \
                --fix_random_seed=False
```

多卡训练：

``` bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --model_name=STNET \
                --config=./configs/stnet.yaml \
                --log_interval=10 \
                --valid_interval=1 \
                --use_gpu=True \
                --save_dir=./data/checkpoints \
                --fix_random_seed=False
```

CPU训练：

``` bash
python train.py --model_name=STNET \
                --config=./configs/stnet.yaml \
                --log_interval=10 \
                --valid_interval=1 \
                --use_gpu=False \
                --save_dir=./data/checkpoints \
                --fix_random_seed=False
```

视频模型库同时提供了快速训练脚本，run.sh，可通过如下命令启动训练:

``` bash
bash run.sh train STNET ./configs/stnet.yaml
```

多卡分布式训练 + GPU视频解码和预处理（仅限TSN模型）

``` bash
bash run_dist.sh train TSN ./configs/tsn_dist_and_dali.yaml
```

- 请根据`CUDA_VISIBLE_DEVICES`指定卡数修改`config`文件中的`num_gpus`和`batch_size`配置。

- 使用CPU训练时请在run.sh中设置use\_gpu=False，使用GPU训练时则设置use\_gpu=True

- 上述启动脚本run.sh运行时需要指定任务类型、模型名、配置文件。训练、评估和预测对应的任务类型分别是train，eval和predict。模型名称则是[AttentionCluster, AttentionLSTM, NEXTVLAD, NONLOCAL, STNET, TSN, TSM, CTCN]中的任何一个。配置文件全部在PaddleVideo/configs目录下，根据模型名称选择对应的配置文件即可。具体使用请参见各模型的说明文档。

- 目前针对TSN模型，做了GPU解码和数据预处理的优化，能明显提升训练速度，具体请参考[TSN](./models/tsn/README.md)

## 模型库结构

### 代码结构

```
configs/
  stnet.yaml
  tsn.yaml
  ...
data/
  dataset/
    youtube/
    kinetics/
    ...
  checkpoints/
    ...
  evaluate_results/
    ...
  predict_results/
    ...
  inference_model/
    ...
reader/
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
utils/
  ...
train.py
eval.py
predict.py
run.sh
```

- `configs`: 各模型配置文件模板
- `reader`: 提供Youtube-8M，Kinetics数据集通用reader，以及模型自定义reader，如nonlocal、ctcn等
- `metrics`: Youtube-8，Kinetics数据集评估脚本，以及模型自定义评估方法
- `models`: 各模型网络结构构建脚本
- `train.py`: 一键式训练脚本，可通过指定模型名，配置文件等一键式启动训练
- `eval.py`: 一键式评估脚本，可通过指定模型名，配置文件，模型权重等一键式启动评估
- `predict.py`: 一键式推断脚本，可通过指定模型名，配置文件，模型权重，待推断文件列表等一键式启动推断
- `run.sh`: 各模型快速训练评估脚本

## Model Zoo

- 基于Youtube-8M数据集模型：

| 模型 | Batch Size | 环境配置 | cuDNN版本 | GAP | 下载链接 |
| :-------: | :---: | :---------: | :-----: | :----: | :----------: |
| Attention Cluster | 2048 | 8卡P40 | 7.1 | 0.84 | [model](https://paddlemodels.bj.bcebos.com/video_classification/AttentionCluster.pdparams) |
| Attention LSTM | 1024 | 8卡P40 | 7.1 | 0.86 | [model](https://paddlemodels.bj.bcebos.com/video_classification/AttentionLSTM.pdparams) |
| NeXtVLAD | 160 | 4卡P40 | 7.1 | 0.87 | [model](https://paddlemodels.bj.bcebos.com/video_classification/NEXTVLAD.pdparams) |

- 基于Kinetics数据集模型：

| 模型 | Batch Size | 环境配置 | cuDNN版本 | Top-1 | 下载链接 |
| :-------: | :---: | :---------: | :----: | :----: | :----------: |
| StNet | 128 | 8卡P40 | 7.1 | 0.69 | [model](https://paddlemodels.bj.bcebos.com/video_classification/STNET.pdparams) |
| TSN | 256 | 8卡P40 | 7.1 | 0.67 | [model](https://paddlemodels.bj.bcebos.com/video_classification/TSN.pdparams) |
| TSM | 128 | 8卡P40 | 7.1 | 0.70 | [model](https://paddlemodels.bj.bcebos.com/video_classification/TSM.pdparams) |
| Non-local | 64 | 8卡P40 | 7.1 | 0.74 | [model](https://paddlemodels.bj.bcebos.com/video_classification/NONLOCAL.pdparams) |

- 基于ActivityNet的动作定位模型：

| 模型 | Batch Size | 环境配置 | cuDNN版本 | 精度 | 下载链接 |
| :-------: | :---: | :---------: | :----: | :----: | :----------: |
| C-TCN | 16 | 8卡P40 | 7.1 | 0.31 (MAP) | [model](https://paddlemodels.bj.bcebos.com/video_detection/CTCN.pdparams) |
| BSN | 16 | 1卡K40 | 7.0 | 66.64% (AUC) | [model-tem](https://paddlemodels.bj.bcebos.com/video_detection/BsnTem.pdparams), [model-pem](https://paddlemodels.bj.bcebos.com/video_detection/BsnPem.pdparams) |
| BMN | 16 | 4卡K40 | 7.0 | 67.19% (AUC) | [model](https://paddlemodels.bj.bcebos.com/video_detection/BMN.pdparams) |

- 基于ActivityNet Captions的视频描述模型:

| 模型 | Batch Size | 环境配置 | cuDNN版本 | METEOR | 下载链接 |
| :-------: | :---: | :---------: | :----: | :----: | :----------: |
| ETS | 256 | 4卡P40 | 7.0 | 9.8 | [model](https://paddlemodels.bj.bcebos.com/video_caption/ETS.pdparams) |

- 基于TACoS的视频查找模型:

| 模型 | Batch Size | 环境配置 | cuDNN版本 | R1@IOU5 | R5@IOU5 | 下载链接 |
| :-------: | :---: | :---------: | :----: | :----: | :----: | :----------: |
| TALL | 56 | 1卡P40 | 7.2 | 0.13 | 0.24 | [model](https://paddlemodels.bj.bcebos.com/video_grounding/TALL.pdparams) |



## 参考文献

- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
- [Beyond Short Snippets: Deep Networks for Video Classification](https://arxiv.org/abs/1503.08909) Joe Yue-Hei Ng, Matthew Hausknecht, Sudheendra Vijayanarasimhan, Oriol Vinyals, Rajat Monga, George Toderici
- [NeXtVLAD: An Efficient Neural Network to Aggregate Frame-level Features for Large-scale Video Classification](https://arxiv.org/abs/1811.05014), Rongcheng Lin, Jing Xiao, Jianping Fan
- [StNet:Local and Global Spatial-Temporal Modeling for Human Action Recognition](https://arxiv.org/abs/1811.01549), Dongliang He, Zhichao Zhou, Chuang Gan, Fu Li, Xiao Liu, Yandong Li, Limin Wang, Shilei Wen
- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool
- [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383v1), Ji Lin, Chuang Gan, Song Han
- [Non-local Neural Networks](https://arxiv.org/abs/1711.07971v1), Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He
- [Bsn: Boundary sensitive network for temporal action proposal generation](http://arxiv.org/abs/1806.02964), Tianwei Lin, Xu Zhao, Haisheng Su, Chongjing Wang, Ming Yang.
- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.
- [Describing Videos by Exploiting Temporal Structure](https://arxiv.org/abs/1502.08029), Li Yao, Atousa Torabi, Kyunghyun Cho, Nicolas Ballas, Christopher Pal, Hugo Larochelle, Aaron Courville.
- [TALL: Temporal Activity Localization via Language Query](https://arxiv.org/abs/1705.02101), Jiyang Gao, Chen Sun, Zhenheng Yang, Ram Nevatia.

## 版本更新

- 3/2019: 新增模型库，发布Attention Cluster，Attention LSTM，NeXtVLAD，StNet，TSN五个视频分类模型。
- 4/2019: 发布Non-local, TSM两个视频分类模型。
- 6/2019: 发布C-TCN视频动作定位模型；Non-local模型增加C2D ResNet101和I3D ResNet50骨干网络；NeXtVLAD、TSM模型速度和显存优化。
- 10/2019: 发布视频动作定位模型BSN, BMN；视频描述模型ETS；视频查找模型TALL。
