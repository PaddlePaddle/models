# TSN 视频分类模型
本目录下为基于PaddlePaddle 动态图实现的TSN视频分类模型。模型支持PaddlePaddle Fluid 1.8, GPU, Linux。

---
## 内容

- [模型简介](#模型简介)
- [安装说明](#安装说明)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [参考论文](#参考论文)


## 模型简介

Temporal Segment Network (TSN) 是视频分类领域经典的基于2D-CNN的解决方案。该方法主要解决视频的长时间行为判断问题，通过稀疏采样视频帧的方式代替稠密采样，既能捕获视频全局信息，也能去除冗余，降低计算量。最终将每帧特征平均融合后得到视频的整体特征，并用于分类。本代码实现的模型为基于单路RGB图像的TSN网络结构，Backbone采用ResNet50结构。

详细内容请参考ECCV 2016年论文[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)

## 安装说明

 ### 环境依赖：

```
    python=3.7
    paddlepaddle-gpu==1.8.3.post97
    opencv=4.3
    CUDA >= 9.0
    cudnn >= 7.5
    wget
    numpy
```

 ### 依赖安装：

 - 安装PaddlePaddle，GPU版本:

    ``` pip3 install paddlepaddle-gpu==1.8.3.post97 -i https://mirror.baidu.com/pypi/simple```
 - 安装opencv 4.2:

    ``` pip3 install opencv-python==4.3.0.36```
 - 安装wget

    ``` pip3 install wget```
 - 安装numpy

    ``` pip3 install numpy```

## 数据准备

TSN的训练数据采用UCF101动作识别数据集。数据下载及准备请参考[数据说明](./data/dataset/ucf101/README.md)

## 模型训练

数据准备完毕后，可以通过如下两种方式启动训练

1. 多卡训练
```bash
bash multi_gpus_run.sh
```
多卡训练所使用的gpu可以通过如下方式设置：
- 修改`multi_gpus_run.sh` 中 `export CUDA_VISIBLE_DEVICES=0,1,2,3`（默认为0,1,2,3表示使用0，1，2，3卡号的gpu进行训练）
- 注意：多卡训练的参数配置文件为`multi_tsn.yaml`。若修改了batchsize则学习率也要做相应的修改，规则为大batchsize用大lr，即同倍数增长缩小关系。例如，默认四卡batchsize=128，lr=0.001，若batchsize=64，lr=0.0005。


2. 单卡训练
```bash
bash single_gpu_run.sh
```
单卡训练所使用的gpu可以通过如下方式设置：
- 修改 `single_gpu_run.sh` 中的 `export CUDA_VISIBLE_DEVICES=0` （表示使用gpu 0 进行模型训练）
- 注意：单卡训练的参数配置文件为`single_gpu_run.sh`。若修改了batchsize则学习率也要做相应的修改，规则为大batchsize用大lr，即同倍数增长缩小关系。默认单卡batchsize=64，lr=0.0005；若batchsize=32，lr=0.00025
## 模型评估

可通过如下方式进行模型评估:
```bash
bash run_eval.sh ./configs/tsn_test.yaml ./weights/final.pdparams
```

- 使用`run.sh`进行评估时，需要修改脚本中的`weights`参数指定需要评估的权重

- `./tsn_test.yaml` 是评估模型时所用的参数文件；`./weights/final.pdparams` 为模型训练完成后，保存的模型文件

- 评估结果以log的形式直接打印输出TOP1\_ACC、TOP5\_ACC等精度指标



实验结果，采用四卡训练，默认配置参数时，在UCF101数据的validation数据集下评估精度如下:

|  | seg\_num | Top-1 | Top-5 |
| :------: | :----------: | :----: | :----: |
| Paddle TSN (静态图) | 3 | 84.00% | 97.38% |
| Paddle TSN (动态图) | 3 | 84.27% | 97.27% |

## 参考论文

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool
