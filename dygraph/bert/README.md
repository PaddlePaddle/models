# BERT on PaddlePaddle

[BERT](https://arxiv.org/abs/1810.04805) 是一个迁移能力很强的通用语义表示模型， 以 [Transformer](https://arxiv.org/abs/1706.03762) 为网络基本组件，以双向 `Masked Language Model`  
和 `Next Sentence Prediction` 为训练目标，通过预训练得到通用语义表示，再结合简单的输出层，应用到下游的 NLP 任务，在多个任务上取得了 SOTA 的结果。本项目是 BERT 在 Paddle Fluid 上的开源实现。

同时推荐用户参考[ IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/122282)

### 发布要点


1）动态图BERT模型，目前仅支持fine-tuning任务，后续会开展对pre-training任务的支持

2）数据集目前验证了glue上的部分任务，squad上的任务后续会进行验证

3）目前暂不支持FP16/FP32混合精度训练。

| Model | Layers | Hidden size | Heads |Parameters |
| :------| :------: | :------: |:------: |:------: |
| [BERT-Base, Uncased](https://baidu-nlp.bj.bcebos.com/DYGRAPH_models/BERT/data.tar.gz) | 12 | 768 |12 |110M |

每个压缩包都包含了模型配置文件 `bert_config.json`、参数文件夹 `params`、动态图参数文件夹`dygraph_params` 和词汇表 `vocab.txt`；

## 内容速览
- [**安装**](#安装)
- [**Fine-Tuning**: 预训练模型如何应用到特定 NLP 任务上](#nlp-任务的-fine-tuning)
  - [语句和句对分类任务](#语句和句对分类任务)

## 目录结构
```text
.
├── data                                        # 示例数据
├── model                                       # 模型定义
├── reader                                      # 数据读取
├── utils                                       # 辅助文件
├── batching.py                                 # 构建 batch 脚本
├── optimization.py                             # 优化方法定义
|── run_classifier.py                           # 分类任务的 fine tuning
|── tokenization.py                             # 原始文本的 token 化
|── train.py                                    # 预训练过程的定义
|── run_classifier_multi_gpu.sh                 # 预训练任务的启动脚本
|── run_classifier_single_gpu.sh                # 预训练任务的启动脚本
```

## 安装
本项目依赖于 Paddle Fluid **1.7.0** 及以上版本，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装。

## NLP 任务的 Fine-tuning

在完成 BERT 模型的预训练后，即可利用预训练参数在特定的 NLP 任务上做 Fine-tuning。以下利用开源的预训练模型，示例如何进行分类任务和阅读理解任务的 Fine-tuning，如果要运行这些任务，请通过 [发布要点](#发布要点) 一节提供的链接预先下载好对应的预训练模型。

### 语句和句对分类任务

对于 [GLUE 数据](https://gluebenchmark.com/tasks)，请下载[文件](https://baidu-nlp.bj.bcebos.com/DYGRAPH_models/BERT/data.tar.gz)，并解压到同一个目录。以 GLUE/MNLI 任务为例，启动 Fine-tuning 的方式如下（也可以直接运行run_classifier_single_gpu.sh）：

```shell
#!/bin/bash

BERT_BASE_PATH="./data/pretrained_models/uncased_L-12_H-768_A-12/"
TASK_NAME='MNLI'
DATA_PATH="./data/glue_data/MNLI/"
CKPT_PATH="./data/saved_model/mnli_models"

export CUDA_VISIBLE_DEVICES=0

# start fine-tuning
python run_classifier.py\
    --task_name ${TASK_NAME} \
    --use_cuda true \
    --do_train true \
    --do_test true \
    --batch_size 64 \
    --init_pretraining_params ${BERT_BASE_PATH}/dygraph_params/ \
    --data_dir ${DATA_PATH} \
    --vocab_path ${BERT_BASE_PATH}/vocab.txt \
    --checkpoints ${CKPT_PATH} \
    --save_steps 1000 \
    --weight_decay  0.01 \
    --warmup_proportion 0.1 \
    --validation_steps 100 \
    --epoch 3 \
    --max_seq_len 128 \
    --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
    --learning_rate 5e-5 \
    --skip_steps 10 \
    --shuffle true

```

这里的 `uncased_L-12_H-768_A-12/` 即是转换后的英文预训练模型，程序会将模型存储在`CKPT_PATH`指定的位置里。

### 使用单机多卡进行fine-tuning

飞桨动态图使用多进程方式进行数据并行和梯度同步，可以参考`run_classifier_multi_gpu.sh`脚本进行单机多卡fine-tuning：

```shell
#!/bin/bash

BERT_BASE_PATH="./data/pretrained_models/uncased_L-12_H-768_A-12/"
TASK_NAME='MNLI'
DATA_PATH="./data/glue_data/MNLI/"
CKPT_PATH="./data/saved_model/mnli_models"
GPU_TO_USE=0,1,2,3

# start fine-tuning
python -m paddle.distributed.launch --selected_gpus=$GPU_TO_USE --log_dir ./cls_log run_classifier.py \
    --task_name ${TASK_NAME} \
    --use_cuda true \
    --use_data_parallel true \
    --do_train true \
    --do_test true \
    --batch_size 64 \
    --in_tokens false \
    --init_pretraining_params ${BERT_BASE_PATH}/dygraph_params/ \
    --data_dir ${DATA_PATH} \
    --vocab_path ${BERT_BASE_PATH}/vocab.txt \
    --checkpoints ${CKPT_PATH} \
    --save_steps 1000 \
    --weight_decay  0.01 \
    --warmup_proportion 0.1 \
    --validation_steps 100 \
    --epoch 3 \
    --max_seq_len 128 \
    --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
    --learning_rate 5e-5 \
    --skip_steps 10 \
    --shuffle true
```

### 读取训练好的模型进行预测

可以参考`run_classifier_prediction.sh`脚本，读取训练好的模型进行预测，可参考以下命令：

```shell
#!/bin/bash

BERT_BASE_PATH="./data/pretrained_models/uncased_L-12_H-768_A-12/"
TASK_NAME='MNLI'
DATA_PATH="./data/glue_data/MNLI/"
CKPT_PATH="./data/saved_model/mnli_models"

export CUDA_VISIBLE_DEVICES=0

# start testing
python run_classifier.py\
    --task_name ${TASK_NAME} \
    --use_cuda true \
    --do_train false \
    --do_test true \
    --batch_size 64 \
    --in_tokens false \
    --data_dir ${DATA_PATH} \
    --vocab_path ${BERT_BASE_PATH}/vocab.txt \
    --checkpoints ${CKPT_PATH} \
    --save_steps 1000 \
    --weight_decay  0.01 \
    --warmup_proportion 0.1 \
    --validation_steps 100 \
    --epoch 3 \
    --max_seq_len 128 \
    --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
    --learning_rate 5e-5 \
    --skip_steps 10 \
    --shuffle false
```
