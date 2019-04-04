# TSM 视频分类模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型推断](#模型推断)
- [参考论文](#参考论文)


## 模型简介

TSM(Temporal Shift Module)，Backbone采用ResNet-50结构。

详细内容请参考论文[Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383)

## 数据准备

TSM的训练数据采用由DeepMind公布的Kinetics-400动作识别数据集。数据下载及准备请参考[数据说明](../../dataset/README.md)

## 模型训练

数据准备完毕后，可以通过如下两种方式启动训练：

    python train.py --model-name=TSM
            --config=./configs/tsm.txt
            --save-dir=checkpoints 
            --log-interval=10 
            --valid-interval=1

    bash scripts/train/train_tsm,.sh

- 可下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/tsm_kinetics.tar.gz)通过`--resume`指定权重存放路径进行finetune等开发

**数据读取器说明：** 模型读取Kinetics-400数据集中的`mp4`数据，每条数据抽取`seg_num`段，每段抽取1帧图像，对每帧图像做随机增强后，缩放至`target_size`。

**训练策略：**

*  采用Momentum优化算法训练，momentum=0.9
*  权重衰减系数为1e-4

## 模型评估

可通过如下两种方式进行模型评估:

    python test.py --model-name=TSM
            --config=configs/tsm.txt
            --log-interval=1
            --weights=$PATH_TO_WEIGHTS

    bash scripts/test/test_tsm.sh

- 使用`scripts/test/test_tsm.sh`进行评估时，需要修改脚本中的`--weights`参数指定需要评估的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/tsm_kinetics.tar.gz)进行评估

当取如下参数时，在Kinetics400的validation数据集下评估精度如下:

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 8 | 224 | 0.70 |

## 模型推断

可通过如下命令进行模型推断：

    python infer.py --model-name=TSM
            --config=configs/tsm.txt
            --log-interval=1 
            --weights=$PATH_TO_WEIGHTS 
            --filelist=$FILELIST

- 模型推断结果存储于`TSM_infer_result`中，通过`pickle`格式存储。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/tsm_kinetics.tar.gz)进行推断

## 参考论文

- [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383), Ji Lin, Chuang Gan, Song Han

