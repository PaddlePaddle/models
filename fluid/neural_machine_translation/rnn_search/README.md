运行本目录下的范例模型需要安装PaddlePaddle Fluid 1.0版。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)中的说明更新 PaddlePaddle 安装版本。

## 机器翻译：RNN Search

以下是本范例模型的简要目录结构及说明：(除下列外，其他文件为CE相关脚本，请无视。)

```text
.
├── README.md              # 文档，本文件
├── args.py                # 训练、预测以及模型参数配置
├── train.py               # 训练主程序
├── infer.py               # 预测主程序
├── attention_model.py     # 带注意力机制的翻译模型配置
└── no_attention_model.py  # 无注意力机制的翻译模型配置
```

### 简介
机器翻译（machine translation, MT）是用计算机来实现不同语言之间翻译的技术。被翻译的语言通常称为源语言（source language），翻译成的结果语言称为目标语言（target language）。机器翻译即实现从源语言到目标语言转换的过程，是自然语言处理的重要研究领域之一。

近年来，深度学习技术的发展不断为机器翻译任务带来新的突破。直接用神经网络将源语言映射到目标语言，即端到端的神经网络机器翻译（End-to-End Neural Machine Translation, End-to-End NMT）模型逐渐成为主流，此类模型一般简称为，简称为NMT模型。

本目录包含机器翻译模型[RNN Search](https://arxiv.org/pdf/1409.0473.pdf)的Paddle Fluid实现。