#  模型列表

## bert-large-cased

| 模型名称 | 模型介绍 | 模型大小  | 模型下载 |
| --- | --- | --- | --- |
|bert-large-cased|  | 1024.00MB | [model_config.json](https://bj.bcebos.com/paddlenlp/models/community/bert-large-cased/model_config.json)<br>[model_state.pdparams](https://bj.bcebos.com/paddlenlp/models/community/bert-large-cased/model_state.pdparams)<br>[tokenizer_config.json](https://bj.bcebos.com/paddlenlp/models/community/bert-large-cased/tokenizer_config.json)<br>[vocab.txt](https://bj.bcebos.com/paddlenlp/models/community/bert-large-cased/vocab.txt) |

也可以通过`paddlenlp` cli 工具来下载对应的模型权重，使用步骤如下所示：

* 安装paddlenlp

```shell
pip install --upgrade paddlenlp
```

* 下载命令行

```shell
paddlenlp download --cache-dir ./pretrained_models bert-large-cased
```

有任何下载的问题都可以到[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)中发Issue提问。