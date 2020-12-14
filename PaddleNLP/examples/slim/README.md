# PaddleSlim-OFA in BERT

BERT-base模型是一个迁移能力很强的通用语义表示模型，但是模型中也有一些参数冗余。本教程将介绍如何使用PaddleSlim对BERT-base模型进行压缩。

## 压缩结果

基于`bert-base-uncased` 在GLUE dev数据集上的finetune结果进行压缩。

| Task  | Metric                       | Result      | Speed Up    |
|-------|------------------------------|-------------|-------------|
| SST-2 | Accuracy                     |        |        |
| QNLI  | Accuracy                     |        |        |
| CoLA  | Mattehew's corr              |        |        |
| MRPC  | F1/Accuracy                  |        |        |
| STS-B | Person/Spearman corr         |        |        |
| QQP   | Accuracy/F1                  |        |        |
| MNLI  | Matched acc/MisMatched acc   |        |        |
| RTE   | Accuracy                     |        |        |

## 快速开始
GLUE/SST-2 任务为例

### Fine-tuing
首先需要对Pretrain-Model在实际的下游任务上进行Finetuning，得到需要压缩的模型。示例以GLUE/SST-2数据集为例。

```shell
cd ../bert/
export PYTHOPATH=${PATH_OF_PaddleNLP}
```

```python
export CUDA_VISIBLE_DEVICES=0
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
参数详细含义参考[README.md](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/examples/bert)
Fine-tuning 在dev上的结果如压缩结果表格中Result那一列所示。

### 安装PaddleSlim
压缩功能依赖最新版本的PaddleSlim.
```shell
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python setup.py install
```

### 压缩训练

```python
python -u ./run_glue_ofa.py --model_type bert \
          --model_name_or_path ${task_pretrained_model_dir} \
          --task_name $TASK_NAME --max_seq_length 128     \
          --batch_size 32       \
          --learning_rate 2e-5     \
          --num_train_epochs 6     \
          --logging_steps 10     \
          --save_steps 100     \
          --output_dir ./tmp/$TASK_NAME \
          --n_gpu 1 \
          --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5
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
- `width_mult_list` 表示压缩训练过程中，对每层Transformer Block的宽度选择的范围。

压缩训练之后在dev上的结果如压缩结果表格中Result with PaddleSlim那一列所示。
