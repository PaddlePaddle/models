#  模型列表

## roberta-base

| 模型名称 | 模型介绍 | 模型大小  | 模型下载 |
| --- | --- | --- | --- |
|roberta-base|  | 625.22MB | [merges.txt](https://bj.bcebos.com/paddlenlp/models/community/roberta-base/merges.txt)<br>[model_config.json](https://bj.bcebos.com/paddlenlp/models/community/roberta-base/model_config.json)<br>[model_state.pdparams](https://bj.bcebos.com/paddlenlp/models/community/roberta-base/model_state.pdparams)<br>[tokenizer_config.json](https://bj.bcebos.com/paddlenlp/models/community/roberta-base/tokenizer_config.json)<br>[vocab.json](https://bj.bcebos.com/paddlenlp/models/community/roberta-base/vocab.json)<br>[vocab.txt](https://bj.bcebos.com/paddlenlp/models/community/roberta-base/vocab.txt) |

也可以通过`paddlenlp` cli 工具来下载对应的模型权重，使用步骤如下所示：

* 安装paddlenlp

```shell
pip install paddlenlp
```

* 下载命令行

```shell
paddlenlp download --cache-dir ./pretrained_models roberta-base
```

有任何下载的问题都可以到[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)中发Issue提问。