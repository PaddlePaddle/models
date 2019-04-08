# NeXtVLAD视频分类模型

---
## 目录

- [算法介绍](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型推断](#模型推断)
- [参考论文](#参考论文)


## 算法介绍
NeXtVLAD模型是第二届Youtube-8M视频理解竞赛中效果最好的单模型，在参数量小于80M的情况下，能得到高于0.87的GAP指标。该模型提供了一种将桢级别的视频特征转化并压缩成特征向量，以适用于大尺寸视频文件的分类的方法。其基本出发点是在NetVLAD模型的基础上，将高维度的特征先进行分组，通过引入attention机制聚合提取时间维度的信息，这样既可以获得较高的准确率，又可以使用更少的参数量。详细内容请参考[NeXtVLAD: An Efficient Neural Network to Aggregate Frame-level Features for Large-scale Video Classification](https://arxiv.org/abs/1811.05014)。

这里实现了论文中的单模型结构，使用2nd-Youtube-8M的train数据集作为训练集，在val数据集上做测试。

## 数据准备

NeXtVLAD模型使用2nd-Youtube-8M数据集, 数据下载及准备请参考[数据说明](../../dataset/README.md)

## 模型训练

### 随机初始化开始训练
在video目录下运行如下脚本即可

    bash ./scripts/train/train_nextvlad.sh

### 使用预训练模型做finetune

请先将提供的预训练模型[model](https://paddlemodels.bj.bcebos.com/video_classification/nextvlad_youtube8m.tar.gz)下载到本地，并在上述脚本文件中添加--resume为所保存的预模型存放路径。

使用4卡Nvidia Tesla P40，总的batch size数是160。

### 训练策略

*  使用Adam优化器，初始learning\_rate=0.0002
*  每2,000,000个样本做一次学习率衰减，learning\_rate\_decay = 0.8
*  正则化使用l2\_weight\_decay = 1e-5

## 模型评估

用户可以下载的预训练模型参数，或者使用自己训练好的模型参数，请在./scripts/test/test\_nextvald.sh
文件中修改--weights参数为保存模型参数的目录。运行

    bash ./scripts/test/test_nextvlad.sh

由于youtube-8m提供的数据中test数据集是没有ground truth标签的，所以这里使用validation数据集来做测试。

模型参数列表如下：

| 参数 | 取值 |
| :---------: | :----: |
| cluster\_size | 128 |
| hidden\_size | 2048 |
| groups | 8 |
| expansion | 2 |
| drop\_rate | 0.5 |
| gating\_reduction | 8 |

计算指标列表如下：

| 精度指标 | 模型精度 |
| :---------: | :----: |
| Hit@1 | 0.8960 |
| PERR | 0.8132 |
| GAP | 0.8709 |

## 模型推断

用户可以下载的预训练模型参数，或者使用自己训练好的模型参数，请在./scripts/infer/infer\_nextvald.sh
文件中修改--weights参数为保存模型参数的目录，运行如下脚本

    bash ./scripts/infer/infer_nextvald.sh

推断结果会保存在NEXTVLAD\_infer\_result文件中，通过pickle格式存储。

## 参考论文

- [NeXtVLAD: An Efficient Neural Network to Aggregate Frame-level Features for Large-scale Video Classification](https://arxiv.org/abs/1811.05014), Rongcheng Lin, Jing Xiao, Jianping Fan

