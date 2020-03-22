# Attention Cluster 视频分类模型

---
## 目录

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型推断](#模型推断)
- [参考论文](#参考论文)


## 模型简介

Attention Cluster模型为ActivityNet Kinetics Challenge 2017中最佳序列模型。该模型通过带Shifting Opeation的Attention Clusters处理已抽取好的RGB、Flow、Audio特征数据，Attention Cluster结构如下图所示。

<p align="center">
<img src="../../images/attention_cluster.png" height=300 width=400 hspace='10'/> <br />
Multimodal Attention Cluster with Shifting Operation
</p>

Shifting Operation通过对每一个attention单元的输出添加一个独立可学习的线性变换处理后进行L2-normalization，使得各attention单元倾向于学习特征的不同成分，从而让Attention Cluster能更好地学习不同分布的数据，提高整个网络的学习表征能力。

详细内容请参考[Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550)

## 数据准备

Attention Cluster模型使用2nd-Youtube-8M数据集, 数据下载及准备请参考[数据说明](../../data/dataset/README.md)

## 模型训练

数据准备完毕后，可以通过如下两种方式启动训练：

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python train.py --model_name=AttentionCluster \
                    --config=./configs/attention_cluster.yaml \
                    --log_interval=10 \
                    --valid_interval=1 \
                    --use_gpu=True \
                    --save_dir=./data/checkpoints \
                    --fix_random_seed=False

    bash run.sh train AttentionCluster ./configs/attention_cluster.yaml

- 可下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/AttentionCluster.pdparams)通过`--resume`指定权重存放路径进行finetune等开发，或者在run.sh脚本中修改resume为解压之后的权重文件存放路径。

**数据读取器说明：** 模型读取Youtube-8M数据集中已抽取好的`rgb`和`audio`数据，对于每个视频的数据，均匀采样100帧，该值由配置文件中的`seg_num`参数指定。

**模型设置：** 模型主要可配置参数为`cluster_nums`和`seg_num`参数，当配置`cluster_nums`为32, `seg_num`为100时，在Nvidia Tesla P40上单卡可跑`batch_size=256`。

**训练策略：**

*  采用Adam优化器，初始learning\_rate=0.001。
*  训练过程中不使用权重衰减。
*  参数主要使用MSRA初始化

## 模型评估

可通过如下两种方式进行模型评估:

    python eval.py --model_name=AttentionCluster \
                   --config=./configs/attention_cluster.yaml \
                   --log_interval=1 \
                   --weights=$PATH_TO_WEIGHTS \
                   --use_gpu=True

    bash run.sh eval AttentionCluster ./configs/attention_cluster.yaml

- 使用`run.sh`进行评估时，需要修改脚本中的`weights`参数指定需要评估的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/AttentionCluster.pdparams)进行评估

- 评估结果以log的形式直接打印输出GAP、Hit@1等精度指标

- 使用CPU进行评估时，请将`use_gpu`设置为False

当取如下参数时:

| 参数 | 取值 |
| :---------: | :----: |
| cluster\_nums | 32 |
| seg\_num | 100 |
| batch\_size | 2048 |
| num\_gpus | 8 |

在2nd-YouTube-8M数据集下评估精度如下:


| 精度指标 | 模型精度 |
| :---------: | :----: |
| Hit@1 | 0.87 |
| PERR | 0.78 |
| GAP | 0.84 |

## 模型推断

可通过如下两种方式启动模型推断：

    python predict.py --model_name=AttentionCluster \
                      --config=configs/attention_cluster.yaml \
                      --log_interval=1 \
                      --weights=$PATH_TO_WEIGHTS \
                      --filelist=$FILELIST \
                      --use_gpu=True

    bash run.sh predict AttentionCluster ./configs/attention_cluster.yaml

- 使用python命令行启动程序时，`--filelist`参数指定待推断的文件列表，如果不设置，默认为data/dataset/youtube8m/infer.list。`--weights`参数为训练好的权重参数，如果不设置，程序会自动下载已训练好的权重。这两个参数如果不设置，请不要写在命令行，将会自动使用默认值。

- 使用`run.sh`进行评估时，请修改脚本中的`weights`参数指定需要用到的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/AttentionCluster.pdparams)进行推断

- 模型推断结果以log的形式直接打印输出，可以看到每个测试样本的分类预测概率。

- 使用CPU进行预测时，请将`use_gpu`设置为False

## 参考论文

- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
