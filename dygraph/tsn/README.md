# TSN 视频分类模型
本目录下为基于PaddlePaddle 动态图实现的TSN视频分类模型。模型支持PaddlePaddle Fluid 2.0, GPU, Linux。

---
## 内容

- [模型简介](#模型简介)
- [安装说明](#安装说明)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [实验结果](#实验结果)
- [参考论文](#参考论文)


## 模型简介

Temporal Segment Network (TSN) 是视频分类领域经典的基于2D-CNN的解决方案。该方法主要解决视频的长时间行为判断问题，通过稀疏采样视频帧的方式代替稠密采样，既能捕获视频全局信息，也能去除冗余，降低计算量。最终将每帧特征平均融合后得到视频的整体特征，并用于分类。本代码实现的模型为基于单路RGB图像的TSN网络结构，Backbone采用ResNet50结构。

详细内容请参考ECCV 2016年论文[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)

## 安装说明

 ### 环境依赖：

```
    python=3.7
    paddlepaddle-gpu==2.0.0a0
    opencv=4.3
    CUDA >= 9.0
    cudnn >= 7.5
    wget
    numpy
```

 ### 依赖安装：

 - 安装PaddlePaddle，GPU版本:

    ``` pip3 install paddlepaddle-gpu==2.0.0a0 -i https://mirror.baidu.com/pypi/simple```
 - 安装opencv 4.3:

    ``` pip3 install opencv-python==4.3.0.36```
 - 安装wget

    ``` pip3 install wget```
 - 安装numpy

    ``` pip3 install numpy```

## 数据准备

TSN的训练数据采用UCF101动作识别数据集。数据下载及处理请参考[数据说明](./data/dataset/ucf101/README.md)。数据处理完成后，会在`./data/dataset/ucf101/`目录下，生成以下文件：
- `videos/` ： 用于存放UCF101数据的视频文件。
- `rawframes/` ： 用于存放UCF101视频文件的frame数据。
- `annotations/` ： 用于存储UCF101数据集的标注文件。
- `ucf101_train_split_{1,2,3}_rawframes.txt`、`ucf101_val_split_{1,2,3}_rawframes.txt`、`ucf101_train_split_{1,2,3}_videos.txt`、`ucf101_val_split_{1,2,3}_videos.txt` ： 为数据的路径list文件。

说明：对应UCF101官方的annotations标注文件，UCF101数据的list文件共有三种不同的切分。例如，ucf101_train_split_1_rawframes.txt 和 ucf101_val_split_1_rawframes.txt 表示对UCF101划分为train和val两部分。ucf101_train_split_2_rawframes.txt 和 ucf101_val_split_2_rawframes.txt 表示对UCF101的另一种train和val划分。训练和测试所使用的list文件，需要一一对应。


## 模型训练
TSN模型训练，需要加载基于imagenet pretrain的ResNet50参数。可通过输入如下命令下载（默认权重文件会存放在当前目前下`./ResNet50_pretrained/`）：
```bash
bash download_pretrain.sh
```

TSN模型支持输入数据为video和frame格式。数据以及预训练参数准备完毕后，可以通过如下方式启动不同格式的训练。

1. 多卡训练（输入为frame格式）
```bash
bash multi_gpus_run.sh ./multi_tsn_frame.yaml
```
多卡训练所使用的gpu可以通过如下方式设置：
- 修改`multi_gpus_run.sh` 中 `export CUDA_VISIBLE_DEVICES=0,1,2,3`（默认为0,1,2,3表示使用0，1，2，3卡号的gpu进行训练）
- 若需要修改预训练权重文件的加载路径，可在`multi_gpus_run.sh`中修改`pretrain`参数（默认`pretrain="./ResNet50_pretrained/"`）
- 注意：多卡、frame格式的训练参数配置文件为`multi_tsn_frame.yaml`。若修改了batchsize则学习率也要做相应的修改，规则为大batchsize用大lr，即同倍数增大缩小关系。例如，默认四卡batchsize=128，lr=0.001，若batchsize=64，lr=0.0005。

2. 多卡训练（输入为video格式）
```bash
bash multi_gpus_run.sh ./multi_tsn_video.yaml
```
多卡训练所使用的gpu可以通过如下方式设置：
- 修改`multi_gpus_run.sh` 中 `export CUDA_VISIBLE_DEVICES=0,1,2,3`（默认为0,1,2,3表示使用0，1，2，3卡号的gpu进行训练）
- 注意：多卡、video格式的训练参数配置文件为`multi_tsn_video.yaml`。若修改了batchsize则学习率也要做相应的修改，规则同上。

3. 单卡训练（输入为frame格式）
```bash
bash single_gpu_run.sh ./single_tsn_frame.yaml
```
单卡训练所使用的gpu可以通过如下方式设置：
- 修改 `single_gpu_run.sh` 中的 `export CUDA_VISIBLE_DEVICES=0` （表示使用gpu 0 进行模型训练）
- 若需要修改预训练权重文件的加载路径，可在`single_gpu_run.sh`中修改`pretrain`参数（默认`pretrain="./ResNet50_pretrained/"`）
- 注意：单卡、frame格式的训练参数配置文件为`single_tsn_frame.yaml`。若修改了batchsize则学习率也要做相应的修改，规则为大batchsize用大lr，即同倍数增长缩小关系。默认单卡batchsize=32，lr=0.00025；若batchsize=64，lr=0.0005。

4. 单卡训练（输入为video格式）
```bash
bash single_gpu_run.sh ./single_tsn_video.yaml
```
单卡训练所使用的gpu可以通过如下方式设置：
- 修改 `single_gpu_run.sh` 中的 `export CUDA_VISIBLE_DEVICES=0` （表示使用gpu 0 进行模型训练）
- 注意：单卡、frame格式的训练参数配置文件为`single_tsn_video.yaml`。若修改了batchsize则学习率也要做相应的修改，规则同上。


## 模型评估

可通过如下方式进行模型评估:
```bash
bash run_eval.sh ./tsn_test.yaml ./weights/final.pdparams
```

- `./tsn_test.yaml` 是评估模型时所用的参数文件；`./weights/final.pdparams` 为模型训练完成后，保存的模型文件

- 评估结果以log的形式直接打印输出TOP1\_ACC、TOP5\_ACC等精度指标



## 实验结果
训练时，Paddle TSN (静态图/动态图) 都才用四卡、输入数据格式为frame, seg_num=3, batchsize=128, lr=0.001。

评估时，输入数据格式为frame，seg_num=25。

备注：seg_num表示训练或者测试时，对每个视频文件采样视频帧的个数。

在UCF101数据validation数据集的评估精度如下:

|  | 路径文件 | seg\_num（训练） | seg\_num（测试）| Top-1 | Top-5 |
| :------: | :----------:| :----------: | :----------: | :----: | :----: |
| Paddle TSN (静态图)|  ucf101_{train/val}_split_1_rawframes.txt|  3  | 25 | 84.00% | 97.38% |
| Paddle TSN (动态图)|  ucf101_{train/val}_split_1_rawframes.txt|  3 | 25 | 84.27% | 97.27% |

## 参考论文

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool
