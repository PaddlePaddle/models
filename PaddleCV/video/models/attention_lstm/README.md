# AttentionLSTM视频分类模型

---
## 内容

- [模型简介](#简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型推断](#模型推断)
- [参考论文](#参考论文)

## 模型简介

递归神经网络（RNN）常用于序列数据的处理，可建模视频连续多帧的时序信息，在视频分类领域为基础常用方法。该模型采用了双向长短记忆网络（LSTM），将视频的所有帧特征依次编码。与传统方法直接采用LSTM最后一个时刻的输出不同，该模型增加了一个Attention层，每个时刻的隐状态输出都有一个自适应权重，然后线性加权得到最终特征向量。论文中实现的是两层LSTM结构，而本代码实现的是带Attention的双向LSTM，Attention层可参考论文[AttentionCluster](https://arxiv.org/abs/1711.09550)。

详细内容请参考[Beyond Short Snippets: Deep Networks for Video Classification](https://arxiv.org/abs/1503.08909)。

## 数据准备

AttentionLSTM模型使用2nd-Youtube-8M数据集，关于数据部分请参考[数据说明](../../dataset/README.md)

## 模型训练

### 随机初始化开始训练

数据准备完毕后，可以通过如下两种方式启动训练：

    python train.py --model_name=AttentionLSTM
            --config=./configs/attention_lstm.txt
            --save_dir=checkpoints
            --log_interval=10
            --valid_interval=1

    bash scripts/train/train_attention_lstm.sh

- AttentionLSTM模型使用8卡Nvidia Tesla P40来训练的，总的batch size数是1024。

### 使用预训练模型做finetune
请先将提供的[model](https://paddlemodels.bj.bcebos.com/video_classification/attention_lstm_youtube8m.tar.gz)下载到本地，并在上述脚本文件中添加`--resume`为所保存的预模型存放路径。

## 模型评估
可通过如下两种方式进行模型评估:

    python test.py --model_name=AttentionLSTM
            --config=configs/attention_lstm.txt
            --log_interval=1
            --weights=$PATH_TO_WEIGHTS

    bash scripts/test/test_attention_lstm.sh

- 使用`scripts/test/test_attention_LSTM.sh`进行评估时，需要修改脚本中的`--weights`参数指定需要评估的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/attention_lstm_youtube8m.tar.gz)进行评估

模型参数列表如下：

| 参数 | 取值 |
| :---------: | :----: |
| embedding\_size | 512 |
| lstm\_size | 1024 |
| drop\_rate | 0.5 |


计算指标列表如下：

| 精度指标 | 模型精度 |
| :---------: | :----: |
| Hit@1 | 0.8885 |
| PERR | 0.8012 |
| GAP | 0.8594 |


## 模型推断

可通过如下命令进行模型推断：

    python infer.py --model_name=attention_lstm
            --config=configs/attention_lstm.txt
            --log_interval=1
            --weights=$PATH_TO_WEIGHTS
            --filelist=$FILELIST

- 模型推断结果存储于`AttentionLSTM_infer_result`中，通过`pickle`格式存储。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/attention_lstm_youtube8m.tar.gz)进行推断

## 参考论文

- [Beyond Short Snippets: Deep Networks for Video Classification](https://arxiv.org/abs/1503.08909) Joe Yue-Hei Ng, Matthew Hausknecht, Sudheendra Vijayanarasimhan, Oriol Vinyals, Rajat Monga, George Toderici

- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen
