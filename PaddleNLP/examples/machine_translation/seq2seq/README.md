# Seq2Seq with Attention

以下是本范例模型的简要目录结构及说明：

```
.
├── README.md              # 文档，本文件
├── args.py                # 训练、预测以及模型参数配置程序
├── data.py                # 数据读入程序
├── download.py            # 数据下载程序
├── train.py               # 训练主程序
├── predict.py             # 预测主程序
└── seq2seq_attn.py        # 带注意力机制的翻译模型程序
```

## 简介

Sequence to Sequence (Seq2Seq)，使用编码器-解码器（Encoder-Decoder）结构，用编码器将源序列编码成vector，再用解码器将该vector解码为目标序列。Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。

本目录包含Seq2Seq的一个经典样例：机器翻译，带attention机制的翻译模型。Seq2Seq翻译模型，模拟了人类在进行翻译类任务时的行为：先解析源语言，理解其含义，再根据该含义来写出目标语言的语句。更多关于机器翻译的具体原理和数学表达式，我们推荐参考飞桨官网[机器翻译案例](https://www.paddlepaddle.org.cn/documentation/docs/zh/user_guides/nlp_case/machine_translation/README.cn.html)。

运行本目录下的范例模型需要安装PaddlePaddle 2.0-rc版。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](https://www.paddlepaddle.org.cn/#quick-start)中的说明更新 PaddlePaddle 安装版本。


## 模型概览

本模型中，在编码器方面，我们采用了基于LSTM的多层的RNN encoder；在解码器方面，我们使用了带注意力（Attention）机制的RNN decoder，在预测时我们使用柱搜索（beam search）算法来生成翻译的目标语句。

## 数据介绍

本教程使用[IWSLT'15 English-Vietnamese data ](https://nlp.stanford.edu/projects/nmt/)数据集中的英语到越南语的数据作为训练语料，tst2012的数据作为开发集，tst2013的数据作为测试集。

### 数据获取
如果用户在初始化数据集时没有提供路径，数据集会自动下载到`paddlenlp.utils.env.DATA_HOME`的`/machine_translation/IWSLT15/`路径下，例如在linux系统下，默认存储路径是`~/.paddlenlp/datasets/machine_translation/IWSLT15`。

## 模型训练

执行以下命令即可训练带有注意力机制的Seq2Seq机器翻译模型：

```sh
python train.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --use_gpu True \
    --model_path ./attention_models

```

各参数的具体说明请参阅 `args.py` 。训练程序会在每个epoch训练结束之后，save一次模型。

**NOTE:** 如需恢复模型训练，则`init_from_ckpt`只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt=attention_models/5`即可，程序会自动加载模型参数`attention_models/5.pdparams`，也会自动加载优化器状态`attention_models/5.pdopt`。

## 模型预测

训练完成之后，可以使用保存的模型（由 `--init_from_ckpt` 指定）对测试集的数据集进行beam search解码，其中译文数据由 `--infer_target_file` 指定），在linux系统下，默认安装路径为`~/.paddlenlp/datasets/machine_translation/IWSLT15/iwslt15.en-vi/tst2013.vi`，如果您使用的是Windows系统，需要更改下面的路径。预测命令如下：

```sh
python predict.py \
     --num_layers 2 \
     --hidden_size 512 \
     --batch_size 128 \
     --dropout 0.2 \
     --init_scale  0.1 \
     --max_grad_norm 5.0 \
     --init_from_ckpt attention_models/9 \
     --infer_target_file ~/.paddlenlp/datasets/machine_translation/IWSLT15/iwslt15.en-vi/tst2013.vi \
     --infer_output_file infer_output.txt \
     --beam_size 10 \
     --use_gpu True
```

各参数的具体说明请参阅 `args.py` ，注意预测时所用模型超参数需和训练时一致。

## 预测效果评价
使用 [*multi-bleu.perl*](https://github.com/moses-smt/mosesdecoder.git) 工具来评价模型预测的翻译质量，将该工具下载在该项目路径下，然后使用如下的命令，可以看到BLEU指标的结果
（需要注意的是，在windows系统下，可能需要更改文件路径`~/.paddlenlp/datasets/machine_translation/IWSLT15/iwslt15.en-vi/tst2013.vi`）：

```sh
perl mosesdecoder/scripts/generic/multi-bleu.perl ~/.paddlenlp/datasets/machine_translation/IWSLT15/iwslt15.en-vi/tst2013.vi < infer_output.txt
```

取第10个epoch保存的模型进行预测，取beam_size=10。效果如下：

```
tst2013 BLEU: 24.40

```
