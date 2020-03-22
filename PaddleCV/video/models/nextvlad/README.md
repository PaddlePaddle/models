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

NeXtVLAD模型使用2nd-Youtube-8M数据集, 数据下载及准备请参考[数据说明](../../data/dataset/README.md)

## 模型训练

### 随机初始化开始训练

在video目录下可以通过如下两种方式启动训练：

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python train.py --model_name=NEXTVLAD \
                    --config=./configs/nextvlad.yaml \
                    --log_interval=10 \
                    --valid_interval=1 \
                    --use_gpu=True \
                    --save_dir=./data/checkpoints \
                    --fix_random_seed=False

    bash run.sh train NEXTVLAD ./configs/nextvlad.yaml

- 在训练NeXtVLAD模型时使用的是4卡，请修改run.sh中的CUDA\_VISIBLE\_DEVICES=0,1,2,3

### 使用预训练模型做finetune

请先将提供的预训练模型[model](https://paddlemodels.bj.bcebos.com/video_classification/NEXTVLAD.pdparams)下载到本地，并在上述脚本文件中添加--resume为所保存的模型参数存放路径。

使用4卡Nvidia Tesla P40，总的batch size数是160。

### 训练策略

*  使用Adam优化器，初始learning\_rate=0.0002
*  每2,000,000个样本做一次学习率衰减，learning\_rate\_decay = 0.8
*  正则化使用l2\_weight\_decay = 1e-5

## 模型评估

可通过如下两种方式进行模型评估:

    python eval.py --model_name=NEXTVLAD \
                   --config=./configs/nextvlad.yaml \
                   --log_interval=1 \
                   --weights=$PATH_TO_WEIGHTS \
                   --use_gpu=True

    bash run.sh eval NEXTVLAD ./configs/nextvlad.yaml

- 使用`run.sh`进行评估时，需要修改脚本中的`weights`参数指定需要评估的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/NEXTVLAD.pdparams)进行评估

- 评估结果以log的形式直接打印输出GAP、Hit@1等精度指标

- 使用CPU进行评估时，请将`use_gpu`设置为False

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

可通过如下两种方式启动模型推断：

    python predict.py --model_name=NEXTVLAD \
                      --config=configs/nextvlad.yaml \
                      --log_interval=1 \
                      --weights=$PATH_TO_WEIGHTS \
                      --filelist=$FILELIST \
                      --use_gpu=True

    bash run.sh predict NEXTVLAD ./configs/nextvlad.yaml

- 使用python命令行启动程序时，`--filelist`参数指定待推断的文件列表，如果不设置，默认为data/dataset/youtube8m/infer.list。`--weights`参数为训练好的权重参数，如果不设置，程序会自动下载已训练好的权重。这两个参数如果不设置，请不要写在命令行，将会自动使用默
认值。

- 使用`run.sh`进行评估时，请修改脚本中的`weights`参数指定需要用到的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/NEXTVLAD.pdparams)进行推断

- 模型推断结果以log的形式直接打印输出，可以看到每个测试样本的分类预测概率。

- 使用CPU进行预测时，请将`use_gpu`设置为False


## 参考论文

- [NeXtVLAD: An Efficient Neural Network to Aggregate Frame-level Features for Large-scale Video Classification](https://arxiv.org/abs/1811.05014), Rongcheng Lin, Jing Xiao, Jianping Fan
