# 使用Seq2Seq模型完成自动对联

以下是本范例模型的简要目录结构及说明：

```
.
├── README.md              # 文档，本文件
├── args.py                # 训练、预测以及模型参数配置程序
├── data.py                # 数据读入程序
├── train.py               # 训练主程序
├── predict.py             # 预测主程序
└── model.py               # 带注意力机制的对联生成程序
```

## 简介

Sequence to Sequence (Seq2Seq)，使用编码器-解码器（Encoder-Decoder）结构，用编码器将源序列编码成vector，再用解码器将该vector解码为目标序列。Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。

本目录包含Seq2Seq的一个经典样例：自动对联生成，带attention机制的文本生成模型。

运行本目录下的范例模型需要安装PaddlePaddle 2.0-rc0版。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](https://www.paddlepaddle.org.cn/#quick-start)中的说明更新 PaddlePaddle 安装版本。


## 模型概览

本模型中，在编码器方面，我们采用了基于LSTM的多层的RNN encoder；在解码器方面，我们使用了带注意力（Attention）机制的RNN decoder，在预测时我们使用柱搜索（beam search）算法来生对联的下联。


## 数据介绍

本教程使用[couplet数据集](https://bj.bcebos.com/paddlehub-dataset/couplet.tar.gz)数据集作为训练语料，train.tsv作为训练集，dev.tsv数据作为开发集，test.tsv数据作为测试集

数据集会在`CoupletDataset`初始化时自动下载

## 模型训练

执行以下命令即可训练带有注意力机制的Seq2Seq模型：

```sh
python train.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --use_gpu True \
    --model_path ./couplet_models \
    --max_epoch 20
```

各参数的具体说明请参阅 `args.py` 。训练程序会在每个epoch训练结束之后，save一次模型。


## 模型预测

训练完成之后，可以使用保存的模型（由 `--init_from_ckpt` 指定）对测试集进行beam search解码，命令如下：

```sh
python predict.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --init_from_ckpt couplet_models/19 \
    --infer_output_file infer_output.txt \
    --beam_size 10 \
    --use_gpu True

```

各参数的具体说明请参阅 `args.py` ，注意预测时所用模型超参数需和训练时一致。

## 生成对联样例

崖悬风雨骤  月落水云寒

约春章柳下  邀月醉花间

箬笠红尘外  扁舟明月中

书香醉倒窗前月    烛影摇红梦里人

踏雪寻梅求雅趣    临风把酒觅知音

未出南阳天下论    先登北斗汉中书

朱联妙语千秋颂    赤胆忠心万代传

月半举杯圆月下    花间对酒醉花间

挥笔如剑倚麓山豪气干云揽月去   落笔似龙飞沧海龙吟破浪乘风来
