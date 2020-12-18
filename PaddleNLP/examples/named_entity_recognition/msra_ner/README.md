# 使用PaddleNLP运行MSRA-NER

## 1. 简介

MSRA-NER 数据集由微软亚研院发布，其目标是识别文本中具有特定意义的实体，主要包括人名、地名、机构名等。示例如下：

```
海钓比赛地点在厦门与金门之间的海域。    OOOOOOOB-LOCI-LOCOB-LOCI-LOCOOOOOO
这座依山傍水的博物馆由国内一流的设计师主持设计，整个建筑群精美而恢宏。    OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
```

数据集中以特殊字符"\t"分隔文本、标签，以特殊字符"\002"分隔每个字。

## 2. 快速开始

### 2.1 环境配置

- Python >= 3.6

- paddlepaddle >= 2.0.0rc1，安装方式请参考 [快速安装](https://www.paddlepaddle.org.cn/install/quick)。

- paddlenlp >= 2.0.0b, 安装方式：`pip install paddlenlp>=2.0.0b`

### 2.2 启动MSRA-NER任务

```shell
export CUDA_VISIBLE_DEVICES=0

python -u ./run_msra_ner.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp/msra_ner/ \
    --n_gpu 1
```

其中参数释义如下：
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
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
global step 996, epoch: 1, batch: 344, loss: 0.038471, speed: 4.72 step/s
global step 997, epoch: 1, batch: 345, loss: 0.032820, speed: 4.82 step/s
global step 998, epoch: 1, batch: 346, loss: 0.008144, speed: 4.69 step/s
global step 999, epoch: 1, batch: 347, loss: 0.031425, speed: 4.36 step/s
global step 1000, epoch: 1, batch: 348, loss: 0.073151, speed: 4.59 step/s
eval loss: 0.019874, precision: 0.991670, recall: 0.991930, f1: 0.991800
```

使用以上命令进行单卡 Fine-tuning ，在验证集上有如下结果：
 Metric                       | Result      |
------------------------------|-------------|
precision                     | 0.992903    |
recall                        | 0.991823    |
f1                            | 0.992363    |

## 参考

[Microsoft Research Asia Chinese Word-Segmentation Data Set](https://www.microsoft.com/en-us/download/details.aspx?id=52531)
[The third international Chinese language processing bakeoff: Word segmentation and named entity recognition](https://faculty.washington.edu/levow/papers/sighan06.pdf)
