# TSM 视频分类模型

本目录下为基于PaddlePaddle 动态图实现的 TSM视频分类模型，静态图实现请参考[TSM 视频分类模型](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo/models/tsm)

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)


## 模型简介

Temporal Shift Module是由MIT和IBM Watson AI Lab的Ji Lin，Chuang Gan和Song Han等人提出的通过时间位移来提高网络视频理解能力的模块, 详细内容请参考论文[Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383v1)

## 数据准备

TSM的训练数据采用由DeepMind公布的Kinetics-400动作识别数据集。数据下载及准备请参考[数据说明](data/dataset/README.md)

### 小数据集验证

为了便于快速迭代，我们采用了较小的数据集进行动态图训练验证，分别进行了两组实验验证：

1. 其中包括8k大小的训练数据和2k大小的测试数据。
2. 其中包括了十类大小的训练数据和测试数据。

## 模型训练

数据准备完毕后，可以通过如下方式启动训练：

    bash run.sh train

## 模型评估

数据准备完毕后，可以通过如下方式启动训练：

    bash run.sh eval

在从Kinetics400选取的十类的数据集下：

|Top-1|Top-5|
|:-:|:-:|
|76.56%|98.1%|

全量数据集精度
Top-1 0.70
请参考：[静态图](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/PaddleVideo)
