# PLATO2

## 模型简介

构建高质量的开放领域（Open-Domain）的对话机器人，使得它能用自然语言与人自由地交流，这一直是自然语言处理领域终极目标之一。

为了能够简易地构建一个高质量的开放域聊天机器人，本项目在Paddle2.0上实现了Plato2的预测模型，并基于终端实现了简单的人机交互。用户可以通过下载预训练模型快速构建一个开放域聊天机器人。

## 快速开始

### 安装说明

* PaddlePaddle 安装

   本项目依赖于 PaddlePaddle 2.0 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

* PaddleNLP 安装

   ```shell
   pip install paddlenlp
   ```

* 环境依赖

    Python的版本要求 3.6+

    本项目依赖sentencepiece和termcolor，请在运行本项目之前进行安装

    ```shell
    pip install sentencepiece termcolor
    ```

    其它环境依赖请参考 PaddlePaddle [安装说明](https://www.paddlepaddle.org.cn/install/quick/zh/2.0rc-linux-docker) 部分的内容


### 代码结构说明

以下是本项目主要代码结构及说明：

```text
.
├── data
│   ├── spm.model # sentencepiece分词预训练模型
│   └── vocab.txt # 词表文件
├── interaction.py # 交互主程序入口
├── model.py # 模型组网
├── readers
│   ├── dialog_reader.py # 模型输入数据生成
│   ├── nsp_reader.py # 模型输入数据生成
│   └── plato_reader.py # 模型输入数据生成
├── utils
│   ├── __init__.py # 基础函数
│   ├── args.py # 运行参数配置
│   ├── masking.py # mask相关函数
│   └── tokenization.py # 分词相关函数
└── README.md # 说明文档
```

### 数据准备

您可以从以下位置下载预训练模型文件：

* Plato2, 24-layers, 16-heads, 1024-hidden, EN: [预训练模型](https://paddlenlp.bj.bcebos.com/models/transformers/plato2/24L.pdparams)
* Plato2, 32-layers, 32-heads, 2048-hidden, EN: [预训练模型](https://paddlenlp.bj.bcebos.com/models/transformers/plato2/32L.pdparams)

以24层预训练模型为例：

```shell
wget https://paddlenlp.bj.bcebos.com/models/transformers/plato2/24L.pdparams
```

**NOTE:** Plato2网络参数量较大，本项目已在22G显存上测试可以支持24层和32层网络，用户可根据显存自行选择合适层数及预训练模型。

### 人机交互

运行如下命令即可开始与聊天机器人用英语进行简单的对话

```shell
export CUDA_VISIBLE_DEVICES=0
python interaction.py --vocab_path ./data/vocab.txt --spm_model_file ./data/spm.model --num_layers 24 --init_from_ckpt ./24L.pdparams
```

以上参数表示：

* vocab_path：词表文件路径。
* spm_model_file：sentencepiece分词预训练模型路径。
* num_layers：Plato2组网层数。
* init_from_ckpt：Plato2预训练模型路径。
