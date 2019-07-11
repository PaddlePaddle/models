运行本目录下的范例模型需要安装PaddlePaddle Fluid 1.5版。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](https://www.paddlepaddle.org.cn/#quick-start)中的说明更新 PaddlePaddle 安装版本。

# 机器翻译：RNN Search

以下是本范例模型的简要目录结构及说明：

```text
.
├── README.md              # 文档，本文件
├── args.py                # 训练、预测以及模型参数
├── train.py               # 训练主程序
├── infer.py               # 预测主程序
├── run.sh                 # 默认配置的启动脚本
├── infer.sh               # 默认配置的解码脚本
├── attention_model.py     # 带注意力机制的翻译模型配置
└── base_model.py          # 无注意力机制的翻译模型配置

```

## 简介
机器翻译（machine translation, MT）是用计算机来实现不同语言之间翻译的技术。被翻译的语言通常称为源语言（source language），翻译成的结果语言称为目标语言（target language）。机器翻译即实现从源语言到目标语言转换的过程，是自然语言处理的重要研究领域之一。

近年来，深度学习技术的发展不断为机器翻译任务带来新的突破。直接用神经网络将源语言映射到目标语言，即端到端的神经网络机器翻译（End-to-End Neural Machine Translation, End-to-End NMT）模型逐渐成为主流，此类模型一般简称为NMT模型。

本目录包含两个经典的机器翻译模型一个base model（不带attention机制），一个带attention机制的翻译模型 .在现阶段，其表现已被很多新模型（如[Transformer](https://arxiv.org/abs/1706.03762)）超越。但除机器翻译外，该模型是许多序列到序列（sequence to sequence, 以下简称Seq2Seq）类模型的基础，很多解决其他NLP问题的模型均以此模型为基础；因此其在NLP领域具有重要意义，并被广泛用作Baseline.

本目录下此范例模型的实现，旨在展示如何用Paddle Fluid实现一个带有注意力机制（Attention）的RNN模型来解决Seq2Seq类问题，以及如何使用带有Beam Search算法的解码器。如果您仅仅只是需要在机器翻译方面有着较好翻译效果的模型，则建议您参考[Transformer的Paddle Fluid实现](https://github.com/PaddlePaddle/models/tree/develop/fluid/neural_machine_translation/transformer)。

## 模型概览
RNN Search模型使用了经典的编码器-解码器（Encoder-Decoder）的框架结构来解决Seq2Seq类问题。这种方法先用编码器将源序列编码成vector，再用解码器将该vector解码为目标序列。这其实模拟了人类在进行翻译类任务时的行为：先解析源语言，理解其含义，再根据该含义来写出目标语言的语句。编码器和解码器往往都使用RNN来实现。关于此方法的具体原理和数学表达式，可以参考[深度学习101](http://paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/basics/machine_translation/index.html).

本模型中，在编码器方面，我们采用了基于LSTM的多层的encoder；在解码器方面，我们使用了带注意力（Attention）机制的RNN decoder，并同时提供了一个不带注意力机制的解码器实现作为对比；而在预测方面我们使用柱搜索（beam search）算法来生成翻译的目标语句。以下将分别介绍用到的这些方法。

## 数据介绍

本教程使用[IWSLT'15 English-Vietnamese data ](https://nlp.stanford.edu/projects/nmt/)数据集中的英语到越南语的数据作为训练语料，tst2012的数据作为开发集，tst2013的数据作为测试集

### 数据获取
```sh
cd data && sh download_en-vi.sh
```


## 训练模型

`run.sh`包含训练程序的主函数，要使用默认参数开始训练，只需要简单地执行：
```sh
sh run.sh
```

```sh
	python train.py \
        --src_lang en --tar_lang vi \
        --attention True \
        --num_layers 2 \
        --hidden_size 512 \
        --src_vocab_size 17191 \
        --tar_vocab_size 7709 \
        --batch_size 128 \
        --dropout 0.2 \
        --init_scale  0.1 \
        --max_grad_norm 5.0 \
        --train_data_prefix data/en-vi/train \
        --eval_data_prefix data/en-vi/tst2012 \
        --test_data_prefix data/en-vi/tst2013 \
        --vocab_prefix data/en-vi/vocab \
        --use_gpu True

```


训练程序会在每个epoch训练结束之后，save一次模型

当模型训练完成之后， 可以利用infer.py的脚本进行预测，默认使用beam search的方法进行预测，加载第10个epoch的模型进行预测，对test的数据集进行解码
```sh
sh infer.sh
```
如果想预测别的数据文件，只需要将 --infer_file参数进行修改

```sh
	python infer.py \
        --src_lang en --tar_lang vi \
        --num_layers 2 \
        --hidden_size 512 \
        --src_vocab_size 17191 \
        --tar_vocab_size 7709 \
        --batch_size 128 \
        --dropout 0.2 \
        --init_scale  0.1 \
        --max_grad_norm 5.0 \
        --vocab_prefix data/en-vi/vocab \
        --infer_file data/en-vi/tst2013.en \
        --reload_model  model_new/epoch_10/ \
        --use_gpu True

```

## 效果

单个模型 beam_size = 10

```sh
no attention

tst2012 BLEU: 11.58
tst2013 BLEU: 12.20



with attention

tst2012 BLEU: 22.21
tst2013 BLEU: 25.30
```
