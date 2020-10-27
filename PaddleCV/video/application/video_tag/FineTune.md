# 模型微调指南

---
## 内容
参考本文档，您可以使用自己的训练数据在VideoTag预训练模型上进行fine-tune，训练出自己的模型。

文档内容包括:
- [原理解析](#原理解析)
- [对AttentionLSTM模型进行微调](#对AttentionLSTM模型进行微调)
- [对TSN模型进行微调](#对TSN模型进行微调)
- [扩展内容](#扩展内容)
- [参考论文](#参考论文)


## 原理解析
VideoTag采用两阶段建模方式，由两个模型组成: TSN + AttentionLSTM。

Temporal Segment Network (TSN) 是经典的基于2D-CNN的视频分类模型。该模型通过稀疏采样视频帧的方式，在捕获视频时序信息的同时降低了计算量。详细内容请参考论文[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)
AttentionLSTM以视频的特征向量作为输入，采用双向长短时记忆网络（LSTM）对所有帧特征进行编码，并增加Attention层，将每个时刻的隐状态输出与自适应权重线性加权得到最终分类向量。详细内容请参考论文[AttentionCluster](https://arxiv.org/abs/1711.09550)

VideoTag训练时分两个阶段: 第一阶段使用少量视频样本（十万级别）训练大规模视频特征提取模型(TSN)；第二阶段使用千万级数据训练预测器(AttentionLSTM)。

VideoTag预测时也分两个阶段: 第一阶段以视频文件作为输入，经过去除了全连接层以及损失函数层的TSN网络后得到输出特征向量；第二阶段以TSN网络输出的特征向量作为输入，经过AttentionLSTM后得到最终的分类结果。

基于我们的预模型，您可以使用自己的训练数据进行fine-tune:

- [对AttentionLSTM模型进行微调](#对AttentionLSTM模型进行微调)
- [对TSN模型进行微调](#对TSN模型进行微调)


## 对AttentionLSTM模型进行微调
AttentionLSTM以视频特征作为输入，显存占用少，训练速度较TSN更快，因此推荐优先对AttentionLSTM模型进行微调。输入视频首先经过TSN预训练模型提取特征向量，然后将特征向量作为训练输入数据，微调AttentionLSTM模型。

### TSN预模型提取特征向量

#### 数据准备

- 预训练权重下载: 参考[样例代码运行指南-数据准备-预训练权重下载](./Run.md)

- 准备训练数据: 准备好待训练的视频数据，并在video\_tag/data/TsnExtractor.list文件中指定待训练的文件路径，内容格式如下:

```
my_video_path/my_video_file1.mp4
my_video_path/my_video_file2.mp4
...
```

#### 特征提取
特征提取脚本如下:

```
python tsn_extractor.py --model_name=TSN --config=./configs/tsn.yaml --weights=./weights/tsn.pdparams
```

- 通过--weights可指定TSN权重参数的存储路径，默认为video\_tag/weights/tsn.pdparams

- 通过--save\_dir可指定特征向量保存路径，默认为video\_tag/data/tsn\_features，不同输入视频的特征向量提取结果分文件保存在不同的npy文件中，目录形式为:

```
video_tag
  ├──data
    ├──tsn_features
      ├── my_feature_file1.npy
      ├── my_feature_file2.npy
      ...
```
- tsn提取的特征向量维度为```帧数*特征维度```，默认为300 * 2048。

### AttentionLSTM模型Fine-tune

#### 数据准备
VideoTag中的AttentionLSTM以TSN模型提取的特征向量作为输入。在video\_tag/data/dataset/attention\_lstm/train.list文件中指定待训练的文件路径和对应的标签，内容格式如下:

```
my_feature_path/my_feature_file1.npy label1 label2
my_feature_path/my_feature_file2.npy label1
...
```
- 一个输入视频可以有多个标签，标签索引为整型数据，文件名与标签之间、多个标签之间以一个空格分隔；

- 标签索引与标签名称的之间的对应关系以list文件指定，可参考VideoTag用到的label_3396.txt文件构造，行索引对应标签索引;

- 验证集、测试集以及预测数据集的构造方式同训练集类似，仅需要在video\_tag/data/attention\_lstm/目录下对应的list文件中指定相关文件路径/标签即可。

#### 模型训练
使用VideoTag中的AttentionLSTM预模型进行fine-tune训练脚本如下:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --model_name=AttentionLSTM --config=./configs/attention_lstm.yaml --pretrain=./weights/attention_lstm
```

- AttentionLSTM模型默认使用8卡训练，总的batch size数是1024。若使用单卡训练，请修改环境变量，脚本如下:
```
export CUDA_VISIBLE_DEVICES=0
python train.py --model_name=AttentionLSTM --config=./configs/attention_lstm-single.yaml --pretrain=./weights/attention_lstm
```

- 请确保训练样本数大于batch_size数

- 通过--pretrain参数可指定AttentionLSTM预训练模型的路径，默认为./weights/attention\_lstm；

- 模型相关配置写在video_tag/configs/attention\_lstm.yaml文件中，可以方便的调节各项超参数；

- 通过--save_dir参数可指定训练模型参数的保存路径，默认为./data/checkpoints；

#### 模型评估
可用如下方式进行模型评估:
```
python eval.py --model_name=AttentionLSTM --config=./configs/attention_lstm.yaml --weights=./data/checkpoints/AttentionLSTM_epoch9.pdparams
```
- 通过--weights参数可指定评估需要的权重，默认为./data/checkpoints/AttentionLSTM_epoch9.pdparams；

- 评估结果以log的形式直接打印输出GAP、Hit@1等精度指标。

#### 模型推断
可用如下方式进行模型推断:
```
python predict.py --model_name=AttentionLSTM --config=./configs/attention_lstm.yaml --weights=./data/checkpoints/AttentionLSTM_epoch9.pdparams
```

- 通过--weights参数可指定推断需要的权重，默认为./data/checkpoints/AttentionLSTM_epoch9.pdparams；

- 通过--label_file参数指定标签文件，请根据自己的数据修改，默认为./label_3396.txt;

- 预测结果会以日志形式打印出来，同时也保存在json文件中，通过--save_dir参数可指定预测结果保存路径，默认为./data/predict_results/attention_lstm。


## 对TSN模型进行微调
VideoTag中使用的TSN模型以mp4文件为输入，backbone为ResNet101。

### 数据准备

准备好训练视频文件后，在video\_tag/data/dataset/tsn/train.list文件中指定待训练的文件路径和对应的标签即可，内容格式如下:

```
my_video_path/my_video_file1.mp4 label1
my_video_path/my_video_file2.mp4 label2
...
```
- 一个输入视频只能有一个标签，标签索引为整型数据，标签索引与文件名之间以一个空格分隔；

- 验证集、测试集以及预测数据集的构造方式同训练集类似，仅需要在video\_tag/data/dataset/tsn目录下对应的list文件中指定相关文件路径/标签即可。

#### 模型训练
使用VideoTag中的TSN预模型进行fine-tune训练脚本如下:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --model_name=TSN --config=./configs/tsn.yaml --pretrain=./weights/tsn
```

- TSN模型默认使用8卡训练，总的batch size数是256。若使用单卡训练，请修改环境变量，脚本如下:
```
export CUDA_VISIBLE_DEVICES=0
python train.py --model_name=TSN --config=./configs/tsn-single.yaml --pretrain=./weights/tsn
```

- 通过--pretrain参数可指定TSN预训练模型的路径，示例为./weights/tsn；

- 模型相关配置写在video_tag/configs/tsn.yaml文件中，可以方便的调节各项超参数；

- 通过--save_dir参数可指定训练模型参数的保存路径，默认为./data/checkpoints；

#### 模型评估
可用如下方式进行模型评估:
```
python eval.py --model_name=TSN --config=./configs/tsn.yaml --weights=./data/checkpoints/TSN_epoch44.pdparams
```

- 通过--weights参数可指定评估需要的权重，示例为./data/checkpoints/TSN_epoch44.pdparams；

- 评估结果以log的形式直接打印输出TOP1_ACC、TOP5_ACC等精度指标。

#### 模型推断
可用如下方式进行模型推断:
```
python predict.py --model_name=TSN --config=./configs/tsn.yaml --weights=./data/checkpoints/TSN_epoch44.pdparams --save_dir=./data/predict_results/tsn/
```

- 通过--weights参数可指定推断需要的权重，示例为./data/checkpoints/TSN_epoch44.pdparams；

- 通过--label_file参数指定标签文件，请根据自己的数据修改，默认为./label_3396.txt;

- 预测结果会以日志形式打印出来，同时也保存在json文件中，通过--save_dir参数可指定预测结果保存路径，示例为./data/predict_results/tsn。

### 训练加速
TSN模型默认以mp4的视频文件作为输入，训练时需要先对视频文件解码，再将解码后的数据送入网络进行训练，如果视频文件很大，这个过程将会很耗时。

为加速训练，可以先将视频解码成图片，然后保存下来，训练时直接根据索引读取帧图片作为输入，加快训练过程。

- 数据准备: 首先将视频解码，存成帧图片；然后生成帧图片的文件路径列表。实现过程可参考[ucf-101数据准备](../../../../dygraph/tsn/data/dataset/ucf101/README.md)

- 修改配置文件: 修改配置文件./config/tsn.yaml，其中MODEL.format值改为"frames"，不同模式下的filelist值改为对应的帧图片文件list。


## 扩展内容

- 更多关于TSN模型的内容可参考PaddleCV视频库[TSN视频分类模型](../../models/tsn/README.md)。

- 更多关于AttentionLSTM模型的内容可参考PaddleCV视频库[AttentionLSTM视频分类模型](../../models/attention_lstm/README.md)。


## 参考论文

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool

- [Beyond Short Snippets: Deep Networks for Video Classification](https://arxiv.org/abs/1503.08909) Joe Yue-Hei Ng, Matthew Hausknecht, Sudheendra Vijayanarasimhan, Oriol Vinyals, Rajat Monga, George Toderici
