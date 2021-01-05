# Distill Bi-LSTM

## 蒸馏结果（Doing)

| Model             | **SST-2(dev)**     |
| ----------------- | ------------------ |
| bert-based        | 0.930045871559633  |
| Bi-LSTM           | 0.856651376146789  |
| Distilled Bi-LSTM | 0.8612385321100917 |


## 蒸馏实验步骤
### 训练bert finetuning模型
以GLUE的SST-2任务为例

```shell
export TASK_NAME=SST-2

python -u ./run_bert_finetune.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 100 \
    --output_dir ./model/$TASK_NAME/ \
    --n_gpu 1 \

```
训练完成之后后，将训练效果最好的模型保存在`models/SST-2/best_model_610`下，该目录下有`model_config.json`, `model_state.pdparams`, `tokenizer_config.json`及`vocab.txt`文件。

### 训练小模型（可选，用于对比蒸馏效果）
本文下载并使用了用Google News语料进行[预训练的word2vec](https://code.google.com/archive/p/word2vec/)初始化网络的Embedding层。
```shell
python small.py
```

### 训练蒸馏模型
将bert的知识蒸馏到基于BiLSTM的小模型中

```shell
python bert_distill.py
```
