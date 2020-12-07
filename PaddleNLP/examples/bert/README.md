# BERT with PaddleNLP

[BERT](https://arxiv.org/abs/1810.04805) 是一个迁移能力很强的通用语义表示模型， 以 [Transformer](https://arxiv.org/abs/1706.03762) 为网络基本组件，以双向 `Masked Language Model`  和 `Next Sentence Prediction` 为训练目标，通过预训练得到通用语义表示，再结合简单的输出层，应用到下游的 NLP 任务，在多个任务上取得了 SOTA 的结果。本项目是 BERT 在 Paddle 2.0上的开源实现。

### 发布要点

1）动态图BERT模型，支持 Fine-tuning，在 GLUE SST-2 任务上进行了验证。
2）支持 Pre-training。

## NLP 任务的 Fine-tuning

在完成 BERT 模型的预训练后，即可利用预训练参数在特定的 NLP 任务上做 Fine-tuning。以下利用开源的预训练模型，示例如何进行分类任务的 Fine-tuning。

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
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/$TASK_NAME/ \
    --n_gpu 1 \

```

其中参数释义如下：
- `model_type` 指示了模型类型，当前仅支持BERT模型。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `task_name` 表示 Fine-tuning 的任务。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `n_gpu` 表示使用的 GPU 卡数。若希望使用多卡训练，将其设置为指定数目即可；若为0，则使用CPU。

训练过程将按照 `logging_steps` 和 `save_steps` 的设置打印如下日志：

```
global step 996, epoch: 0, batch: 996, loss: 0.248909, speed: 5.07 step/s
global step 997, epoch: 0, batch: 997, loss: 0.113216, speed: 4.53 step/s
global step 998, epoch: 0, batch: 998, loss: 0.218075, speed: 4.55 step/s
global step 999, epoch: 0, batch: 999, loss: 0.133626, speed: 4.51 step/s
global step 1000, epoch: 0, batch: 1000, loss: 0.187652, speed: 4.45 step/s
eval loss: 0.083172, accu: 0.920872
```

使用以上命令进行单卡 Fine-tuning ，在验证集上有如下结果：

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| SST-2 | Accuracy                     | 92.88       |
| QNLI  | Accuracy                     | 91.67       |

## 预训练

```shell
export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR=/guosheng/nv-bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT/data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/

python -u ./run_pretrain.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 1e5 \
    --input_dir $DATA_DIR \
    --output_dir ./tmp2/ \
    --logging_steps 1 \
    --save_steps 20000 \
    --max_steps 1000000 \
    --n_gpu 2
```
