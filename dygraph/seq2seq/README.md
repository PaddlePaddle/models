运行本目录下的范例模型需要安装PaddlePaddle 2.0版。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](https://www.paddlepaddle.org.cn/#quick-start)中的说明更新 PaddlePaddle 安装版本。

# Sequence to Sequence (Seq2Seq)

以下是本范例模型的简要目录结构及说明：

```
.
├── README.md              # 文档，本文件
├── args.py                # 训练、预测以及模型参数配置程序
├── reader.py              # 数据读入程序
├── train.py               # 训练主程序
├── run.sh                 # 默认配置的启动脚本
└──attention_model.py      # 带注意力机制的翻译模型程序
```

## 简介

Sequence to Sequence (Seq2Seq)，使用编码器-解码器（Encoder-Decoder）结构，用编码器将源序列编码成vector，再用解码器将该vector解码为目标序列。Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。

本目录包含Seq2Seq的一个经典样例：机器翻译，实现了一个base model（不带attention机制），一个带attention机制的翻译模型。Seq2Seq翻译模型，模拟了人类在进行翻译类任务时的行为：先解析源语言，理解其含义，再根据该含义来写出目标语言的语句。更多关于机器翻译的具体原理和数学表达式，我们推荐参考[深度学习101](http://paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/basics/machine_translation/index.html)。

**本目录旨在展示如何用PaddlePaddle 2.0-beta的动态图接口实现一个标准的Seq2Seq模型** ，相同网络结构的静态图实现可以参照 [Seq2Seq](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/PaddleTextGEN/seq2seq)。

## 模型概览

本模型中，在编码器方面，我们采用了基于LSTM的多层的RNN encoder；在解码器方面，我们使用了带注意力（Attention）机制的RNN decoder，并同时提供了一个不带注意力机制的解码器实现作为对比。在预测时我们使用柱搜索（beam search）算法来生成翻译的目标语句。以下将分别介绍用到的这些方法。

## 数据介绍

本教程使用[IWSLT'15 English-Vietnamese data ](https://nlp.stanford.edu/projects/nmt/)数据集中的英语到越南语的数据作为训练语料，tst2012的数据作为开发集，tst2013的数据作为测试集

### 数据获取

```
cd ..
python download.py
```

## 模型训练

`run.sh`包含训练程序的主函数，要使用默认参数开始训练，只需要简单地执行：

```
sh run.sh
```


```
python train.py \
    --src_lang en --tar_lang vi \
    --num_layers 2 \
    --hidden_size 512 \
    --src_vocab_size 17191 \
    --tar_vocab_size 7709 \
    --batch_size 128 \
    --dropout 0.0 \
    --init_scale  0.2 \
    --max_grad_norm 5.0 \
    --train_data_prefix data/en-vi/train \
    --eval_data_prefix data/en-vi/tst2012 \
    --test_data_prefix data/en-vi/tst2013 \
    --vocab_prefix data/en-vi/vocab \
    --use_gpu True \
    --model_path attention_models \
    --enable_ce \
    --learning_rate 0.002 \
    --dtype float64 \
    --optimizer sgd \
    --max_epoch 3
```

训练程序会在每个epoch训练结束之后，save一次模型。

## 模型预测

TODO

## 效果评价

使用 [*multi-bleu.perl*](https://github.com/moses-smt/mosesdecoder.git) 工具来评价模型预测的翻译质量，使用方法如下：

```sh
mosesdecoder/scripts/generic/multi-bleu.perl tst2013.vi < infer_output.txt
```
