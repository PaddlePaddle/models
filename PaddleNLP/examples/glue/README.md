# GLUE with PaddleNLP

[GLUE](https://gluebenchmark.com/)是当今使用最为普遍的自然语言理解评测基准数据集，评测数据涵盖新闻、电影、百科等许多领域，其中有简单的句子，也有困难的句子。其目的是通过公开的得分榜，促进自然语言理解系统的发展。详细以参考 [GLUE论文](https://openreview.net/pdf?id=rJ4km2R5t7)

本项目是 GLUE评测任务 在 Paddle 2.0上的开源实现。

## 发布要点

1. 支持CoLA、SST-2、MRPC、STS-B、QQP、MNLI、QNLI、RTE 8个GLUE评测任务的Fine-tuning。
2. 支持 BERT、ELECTRA 等预训练模型运行这些GLUE评测任务。

## NLP 任务的 Fine-tuning
运行Fine-tuning有两种方式：
1. 使用已有的预训练模型运行 Fine-tuning。
2. 运行特定模型（如BERT、ELECTRA等）的预训练后，使用预训练模型运行 Fine-tuning（需要很多资源）。

以下例子基于方式1。

### 语句和句对分类任务

以 GLUE/SST-2 任务为例，启动 Fine-tuning 的方式如下（`paddlenlp` 要已经安装或能在 `PYTHONPATH` 中找到）：

```shell
export CUDA_VISIBLE_DEVICES=0,1
export TASK_NAME=SST-2

python -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/$TASK_NAME/ \
    --n_gpu 1 \

```

其中参数释义如下：
- `model_type` 指示了模型类型，当前支持BERT、ELECTRA模型。
- `model_name_or_path` 指示了使用哪种预训练模型，对应有其预训练模型和预训练时使用的 tokenizer，当前支持bert-base-uncased、bert-large-uncased、bert-base-cased、bert-large-cased、bert-base-multilingual-uncased、bert-base-multilingual-cased、bert-base-chinese、bert-wwm-chinese、bert-wwm-ext-chinese、electra-small、electra-base、electra-large、chinese-electra-base、chinese-electra-small等模型。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `task_name` 表示 Fine-tuning 的任务，当前支持CoLA、SST-2、MRPC、STS-B、QQP、MNLI、QNLI、RTE。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `n_gpu` 表示使用的 GPU 卡数。若希望使用多卡训练，将其设置为指定数目即可；若为0，则使用CPU。

Fine-tuning过程将按照 `logging_steps` 和 `save_steps` 的设置打印如下日志：

```
global step 6310/6315, epoch: 2, batch: 2099, rank_id: 0, loss: 0.035772, lr: 0.0000000880, speed: 3.1527 step/s
global step 6311/6315, epoch: 2, batch: 2100, rank_id: 0, loss: 0.056789, lr: 0.0000000704, speed: 3.4201 step/s
global step 6312/6315, epoch: 2, batch: 2101, rank_id: 0, loss: 0.096717, lr: 0.0000000528, speed: 3.4694 step/s
global step 6313/6315, epoch: 2, batch: 2102, rank_id: 0, loss: 0.044982, lr: 0.0000000352, speed: 3.4513 step/s
global step 6314/6315, epoch: 2, batch: 2103, rank_id: 0, loss: 0.139579, lr: 0.0000000176, speed: 3.4566 step/s
global step 6315/6315, epoch: 2, batch: 2104, rank_id: 0, loss: 0.046043, lr: 0.0000000000, speed: 3.4590 step/s
eval loss: 0.549763, acc: 0.9151376146788991, eval done total : 1.8206987380981445 s
```

使用electra-small预训练模型进行单卡 Fine-tuning ，在验证集上有如下结果：

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthews corr                | 58.22       |
| SST-2 | acc.                         | 91.85       |
| MRPC  | acc./F1                      | 88.24       |
| STS-B | Pearson/Spearman corr        | 87.24       |
| QQP   | acc./F1                      | 88.83       |
| MNLI  | matched acc./mismatched acc. | 82.45       |
| QNLI  | acc.                         | 88.61       |
| RTE   | acc.                         | 66.78       |

注：acc.是Accuracy的简称，表中Metric字段名词取自[GLUE论文](https://openreview.net/pdf?id=rJ4km2R5t7)
