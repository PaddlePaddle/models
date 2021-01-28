# Distill Bi-LSTM
以下是本例的简要目录结构及说明：
```
.
├── small.py              # 小模型结构以及对小模型的训练脚本
├── bert_distill.py       # 用大模型bert蒸馏小模型的蒸馏脚本
├── data.py               # 定义了dataloader等数据读取接口
├── utils.py              # 定义了将样本转成id的转换接口
├── args.py               # 参数配置脚本
└── README.md             # 文档，本文件
```

## 蒸馏结果
利用bert模型去蒸馏基于BiLSTM的小模型，对比小模型单独训练，在SST-2、QQP、senta(中文情感分类)任务上分别有3.2%、1.8%、1.4%的提升。

| Model          | SST-2(dev acc)    | QQP(dev acc/f1)            | ChnSentiCorp(dev acc) | ChnSentiCorp(dev acc) |
| -------------- | ----------------- | -------------------------- | --------------------- | --------------------- |
| teacher  model | bert-base-uncased | bert-base-uncased          | bert-base-chinese     | bert-wwm-ext-chinese  |
| Teacher        | 0.930046          | 0.905813(acc)/0.873472(f1) | 0.951667              | 0.955000              |
| Student        | 0.853211          | 0.856171(acc)/0.806057(f1) | 0.920833              | 0.920800              |
| Distilled      | 0.885321          | 0.874375(acc)/0.829581(f1) | 0.930000              | 0.935000              |


## 蒸馏实验步骤
### 训练bert finetuning模型
以GLUE的SST-2任务为例：

```shell
cd ../../glue
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=SST-2
python -u ./run_bert_finetune.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 128   \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 10 \
    --output_dir ../distill/ditill_lstm/model/$TASK_NAME/ \
    --n_gpu 1 \

```
训练完成之后，将训练效果最好的模型保存在该项目下的`models/$TASK_NAME/`下。该模型目录下有`model_config.json`, `model_state.pdparams`, `tokenizer_config.json`及`vocab.txt`这几个文件。

### 数据准备

在中文数据集上的小模型训练是利用jieba分词的，词表是由senta提供，可通过运行以下命令进行下载
```shell
wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt
```
进行下载，为了节省显存和运行时间，可以对ChnSentiCorp中出现的词先进行过滤，并将词表文件名配置在下面的参数中。

本文下载并在对英文数据集的训练中使用了Google News语料进行[预训练的word2vec](https://code.google.com/archive/p/word2vec/)初始化网络的Embedding层。由于语料大多是英文，因此预训练的Embedding对于中文数据集的模型训练可能没有帮助，因此在中文中Embedding层的参数采用了随机初始化的方式。

### 训练小模型

尝试运行下面的脚本可以运行下面的命令分别基于ChnSentiCorp、SST-2、QQP数据集对基于BiLSTM的小模型进行训练。


```shell
CUDA_VISIBLE_DEVICES="0" nohup python small.py \
    --task_name senta \
    --max_epoch 20 \
    --vocab_size 29496 \
    --batch_size 64 \
    --model_name bert-wwm-ext-chinese \
    --optimizer adam \
    --lr 3e-4 \
    --dropout_prob 0.2 \
    --use_pretrained_emb False \
    --vocab_path senta_word_dict_subset.txt > senta_small.log &
```

```shell
CUDA_VISIBLE_DEVICES="1" nohup python small.py \
    --task_name sst-2 \
    --vocab_size 30522 \
    --max_epoch 10 \
    --batch_size 64 \
    --lr 1.0 \
    --dropout_prob 0.4 \
    --use_pretrained_emb True > sst-2_small.log &
```

```shell
CUDA_VISIBLE_DEVICES="2" nohup python small.py \
    --task_name qqp \
    --vocab_size 30522 \
    --max_epoch 35 \
    --batch_size 256 \
    --lr 2.0 \
    --dropout_prob 0.4 \
    --use_pretrained_emb True > qqp_small.log &
```

### 训练蒸馏模型
这一步是将bert的知识蒸馏到基于BiLSTM的小模型中，可以运行下面的命令分别基于ChnSentiCorp、SST-2、QQP数据集对基于BiLSTM的小模型进行蒸馏。

```shell
CUDA_VISIBLE_DEVICES="3" nohup python bert_distill.py \
    --task_name senta \
    --vocab_size 29496 \
    --max_epoch 6 \
    --lr 1.0 \
    --dropout_prob 0.1 \
    --batch_size 64 \
    --model_name bert-wwm-ext-chinese \
    --use_pretrained_emb False \
    --teacher_path model/senta/best_bert_wwm_ext_model_880/model_state.pdparams \
    --vocab_path senta_word_dict_subset.txt > senta_distill_wwm_ext.log &
```

```shell
CUDA_VISIBLE_DEVICES="4" nohup python bert_distill.py \
    --task_name sst-2 \
    --vocab_size 30522 \
    --max_epoch 6 \
    --lr 1.0 \
    --task_name sst-2 \
    --dropout_prob 0.2 \
    --batch_size 128 \
    --model_name bert-base-uncased \
    --use_pretrained_emb True \
    --teacher_path model/SST-2/best_model_610/model_state.pdparams > sst-2_distill.log &
```

```shell
CUDA_VISIBLE_DEVICES="5" nohup python bert_distill.py \
    --task_name qqp \
    --vocab_size 30522 \
    --max_epoch 6 \
    --lr 1.0 \
    --dropout_prob 0.2 \
    --batch_size 256 \
    --model_name bert-base-uncased \
    --use_pretrained_emb True \
    --n_iter 10 \
    --teacher_path model/QQP/best_model_17000/model_state.pdparams > qqp_distill.log &
```
