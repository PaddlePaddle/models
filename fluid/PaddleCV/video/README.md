# Paddle视频模型库

---

## 安装

在当前模型库运行样例代码需要PadddlePaddle Fluid的v.1.2.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/install/index_cn.html)中的说明来更新PaddlePaddle。

## 简介
本教程期望给开发者提供基于PaddlePaddle的便捷、高效的使用深度学习算法解决视频理解、视频编辑、视频生成等一系列模型。目前我们包含5个视频分类模型，后续我们会不断的扩展到其他更多场景。

目前视频分类模型包括:

| 模型 | 类别  | 描述 |
| :---------------: | :--------: | :------------: |
| [Attention Cluster](./models/attention_cluster/README.md) [[论文](https://arxiv.org/abs/1711.09550)] | 视频分类| CVPR'18提出的视频多模态特征注意力聚簇融合方法 |
| [Attention LSTM](./models/attention_lstm/README.md) [[论文](https://arxiv.org/abs/1503.08909)] | 视频分类| 常用模型，速度快精度高 |
| [NeXtVLAD](./models/nextvlad/README.md) [[论文](https://arxiv.org/abs/1811.05014)] | 视频分类| 2nd-Youtube-8M最优单模型 |
| [StNet](./models/stnet/README.md) [[论文](https://arxiv.org/abs/1811.01549)] | 视频分类| AAAI'19提出的视频联合时空建模方法 |
| [TSN](./models/tsn/README.md) [[论文](https://arxiv.org/abs/1608.00859)] | 视频分类| ECCV'16提出的基于2D-CNN经典解决方案 |


## 数据准备

视频模型库使用Youtube-8M和Kinetics数据集, 具体使用方法请参考[数据说明](./dataset/README.md)

## 快速使用

视频模型库提供通用的train/test/infer框架，通过`train.py/test.py/infer.py`指定模型名、模型配置参数等可一键式进行训练和预测。

视频库目前支持的模型包括：[Attention Cluster](./models/attention_cluster/README.md), [Attention LSTM](./models/attention_lstm/README.md), [NeXtVLAD](./models/nextvlad/README.md), [StNet](./models/stnet/README.md), [TSN](./models/tsn/README.md)。

以AttentionCluster模型为例：

单卡训练：

``` bash
export CUDA_VISIBLE_DEVICES=0
python train.py --model-name=AttentionCluster
        --config=./configs/attention_cluster.txt
        --epoch=8
        --batch-size=256
        --save-dir=checkpoints
        --log-interval=10
        --valid-interval=1
```

多卡训练：

``` bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --model-name=AttentionCluster
        --config=./configs/attention_cluster.txt
        --epoch=8
        --batch-size=2048
        --save-dir=checkpoints
        --log-interval=10
        --valid-interval=1
```

- 请根据`CUDA_VISIBLE_DEVICES`指定卡数修改`config`文件中`num_gpus`和`batch_size`。

## Model Zoo

使用Youtube-8M数据集模型如下：

| 模型 | Batch Size | 环境配置 | cuDNN版本 | GAP | 下载链接 |
| :-------: | :---: | :---------: | :-----: | :----: | :----------: |
| Attention Cluster | 2048 | 8卡P40 | 7.1 | 0.84 | [model](https://paddlemodels.bj.bcebos.com/video_classification/attention_cluster_youtube8m.tar.gz) |
| Attention LSTM | 1024 | 8卡P40 | 7.1 | 0.86 | [model](https://paddlemodels.bj.bcebos.com/video_classification/attention_lstm_youtube8m.tar.gz) |
| NeXtVLAD | 160 | 4卡P40 | 7.1 | 0.87 | [model](https://paddlemodels.bj.bcebos.com/video_classification/nextvlad_youtube8m.tar.gz) |

使用Kinetics数据集模型如下：

| 模型 | Batch Size | 环境配置 | cuDNN版本 | Top-1 | 下载链接 |
| :-------: | :---: | :---------: | :----: | :----: | :----------: |
| StNet | 128 | 8卡P40 | 5.1 | 0.69 | [model](https://paddlemodels.bj.bcebos.com/video_classification/stnet_kinetics.tar.gz) |
| TSN | 256 | 8卡P40 | 7.1 | 0.67 | [model](https://paddlemodels.bj.bcebos.com/video_classification/tsn_kientics.tar.gz) |

## 版本更新

- 3/2019: 新增模型库

