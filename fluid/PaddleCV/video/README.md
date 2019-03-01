# Paddle视频模型库

---

## 安装

在当前模型库运行样例代码需要PadddlePaddle Fluid的v.1.2.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/install/index_cn.html)中的说明来更新PaddlePaddle。

## 简介
本次发布的是Paddle视频模型库第一期，包括五个视频分类模型。后续我们将会扩展到视频理解方向的更多应用场景以及视频编辑和生成等方向，以便为开发者提供简单、便捷的使用深度学习算法处理视频的途径。

Paddle视频模型库第一期主要包含如下模型。

| 模型 | 类别  | 描述 |
| :---------------: | :--------: | :------------: |
| [Attention Cluster](./models/attention_cluster/README.md) | 视频分类| CVPR'18提出的视频多模态特征注意力聚簇融合方法 |
| [Attention LSTM](./models/attention_lstm/README.md) | 视频分类| 常用模型，速度快精度高 |
| [NeXtVLAD](./models/nextvlad/README.md) | 视频分类| 2nd-Youtube-8M最优单模型 |
| [StNet](./models/stnet/README.md) | 视频分类| AAAI'19提出的视频联合时空建模方法 |
| [TSN](./models/tsn/README.md) | 视频分类| 基于2D-CNN经典解决方案 |


## 数据准备

视频模型库使用Youtube-8M和Kinetics数据集, 具体使用方法请参考[数据说明](./dataset/README.md)

## 快速使用

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

详细使用步骤请参考各模型文档：[Attention Cluster](./models/attention_cluster/README.md), [Attention LSTM](./models/attention_lstm/README.md), [NeXtVLAD](./models/nextvlad/README.md), [StNet](./models/stnet/README.md), [TSN](./models/tsn/README.md)

## 模型精度

模型库各模型评估精度如下：

| 模型 | 数据集 | 精度类别  | 精度 |
| :---------------: | :-----------: | :-------: | :------: |
| AttentionCluster | Youtube-8M | GAP | 0.84 |
| AttentionLSTM | Youtube-8M | GAP | 0.86 |
| NeXtVLAD | Youtube=8M | GAP | 0.87 |
| StNet | Kinetics | Top-1 | 0.69 |
| TSN | Kinetics | Top-1 | 0.67 |

## Model Zoo

| 模型 | Batch Size | 环境配置 | CUDA版本 | CUDNN版本 | 下载链接 |
| :-------: | :---: | :---------: | :----: | :-----: | :----------: |
| Attention Cluster | 2048 | 8卡P40 | 8.0 | 7.1 | [model](https://paddlemodels.bj.bcebos.com/video_clasification/attention_cluster_youtube8m.tar.gz) |
| Attention LSTM | 1024 | 8卡P40 | 8.0 | 7.1 | [model](https://paddlemodels.bj.bcebos.com/video_clasification/attention_lstm_youtube8m.tar.gz) |
| NeXtVLAD | 160 | 4卡P40 | 8.0 | 7.1 | [model](https://paddlemodels.bj.bcebos.com/video_clasification/attention_cluster_youtube8m.tar.gz) |
| StNet | 128 | 8卡P40 | 8.0 | 5.1 | [model](https://paddlemodels.bj.bcebos.com/video_clasification/stnet_kientics.tar.gz) |
| TSN | 256 | 8卡P40 | 8.0 | 7.1 | [model](https://paddlemodels.bj.bcebos.com/video_clasification/tsn_kientics.tar.gz) |

