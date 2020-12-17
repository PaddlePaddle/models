简体中文 | [English](./README_en.md)
<p align="center">
  <img src="./docs/imgs/paddlenlp.png" width="520" height ="100"  align="middle" />
</p>

基于PaddlePaddle2.0的NLP领域深度学习框架，包含丰富的前沿模型、易学易用的高层API、覆盖主流任务的全流程、高性能代码实现。

 [![python version](https://camo.githubusercontent.com/4bc45421df57c3901ec5d21da412680df9b2d74fee7c297ab4e6764868e805fb/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e362b2d6f72616e67652e737667)](https://camo.githubusercontent.com/4bc45421df57c3901ec5d21da412680df9b2d74fee7c297ab4e6764868e805fb/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e362b2d6f72616e67652e737667) [![support os](https://camo.githubusercontent.com/7c97d13875070c3d1cfc86838fa87cb3db7909847a3992a33665c0a67800a33a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6f732d6c696e757825324325323077696e2532432532306d61632d79656c6c6f772e737667)](https://camo.githubusercontent.com/7c97d13875070c3d1cfc86838fa87cb3db7909847a3992a33665c0a67800a33a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6f732d6c696e757825324325323077696e2532432532306d61632d79656c6c6f772e737667)



# 特性

- 强大的模型库

  - 涵盖了NLP主流应用相关的前沿模型，包括词法和句法分析、文本向量化表示、文本分类、匹配、生成、机器翻译、通用对话、问答系统...

- 易学易用的高层API

  - 完美继承PaddlePaddle2.0的高层API体系，易学易用，便于开发，在数据处理、数据加载、模型构建、训练和预测等各个环节上，降低代码量，提高开发效率。

- 支持大规模训练，达到极致性能

  - 结合Fleet API，以BERT为例，提供了高性能的分布式训练示范。能够充分利用GPU集群资源，达到极致性能，详见[benchmark](./benchmark/bert) 。

- 结合行业案例的详尽教程，从数据处理到预测部署的全流程实现

  - 提供了结合行业案例的详细教程，以notebook形式展示。

  

# 安装

### 安装依赖

- python >= 3.6
- paddlepaddle >= 2.0.0-rc1

```
pip install paddlenlp==2.0.0b 
```



# 快速开始

## 一键加载数据集

```python
from paddlenlp.datasets import ChnSentiCrop

train_dataset, dev_dataset, test_dataset= ChnSentiCorp.get_datasets(['train', 'dev', 'test'])
```

可参考[Dataset文档](./docs/datasets.md)查看更多数据集。

## 内置中文embedding

```python
from paddlenlp.embeddings import TokenEmbedding

wordemb = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300")
print(wordemb.cosine_sim("国王", "王后"))
>>> 0.63395125
wordemb.cosine_sim("艺术", "火车")
>>> 0.14792643
```

更多使用方法请参考 [word_embedding文档](./examples/word_embedding/README.md)。

## 一键加载复杂模型
```python
from paddlenlp.models import Ernie, Senta, SimNet

ernie = Ernie("ernie-1.0", num_classes=2, task="seq-cls")

senta = Senta(network="bow", vocab_size=1024, num_classes=2)

simnet = SimNet(network="gru", vocab_size=1024, num_classes=2)
```

## 丰富、高质量的中文预训练模型
```python
from paddlenlp.transformers import ErnieModel, BertModel, RobertaModel, ElectraModel

ernie = ErnieModel.from_pretrained('ernie-1.0')

bert = BertModel.from_pretrained('bert-wwm-chinese')

roberta = RobertaModel.from_pretrained('roberta-wwm-ext')

electra = ElectraModel.from_pretrained('chinese-electra-small')
```

请参考 [Pretrained-Models](./docs/transformers.md)查看目前支持的预训练模型。



# API list

- [Transformer API](./docs/transformers.md)

- [Dataset API](./docs/datasets.md)

- [Embedding API](./docs/embeddings.md)

- [Metrics API](./docs/embeddings.md)

- [Models API](./docs/models.md)

  

# 教程

- [使用seq2vec模块进行句子情感分类](https://aistudio.baidu.com/aistudio/projectdetail/1283423)
- [如何将预训练模型Fine-tune下游任务](https://aistudio.baidu.com/aistudio/projectdetail/1294333)
- [使用Bi-GRU+CRF完成快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1317771)
- [使用预训练模型ERNIE优化快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1329361)
- [使用seq2seq模型写对联](https://aistudio.baidu.com/aistudio/projectdetail/1321118)
- [使用预训练模型ERNIE-GEN生成诗歌](https://aistudio.baidu.com/aistudio/projectdetail/1339888)
- [使用PaddleNLP完成新冠疫情病例数预测](https://aistudio.baidu.com/aistudio/projectdetail/1290873)

更多教程参见[PaddleNLP on AI Studio](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)。



# QQ交流群

<img src="./docs/imgs/qq.png" align="left" width="200" height ="200"/>

扫一扫二维码，加入群聊 ⬆️



# 社区贡献

- 欢迎您向PaddleNLP贡献优秀的预训练模型、数据集...
- 您也可以联系我们贡献API文档、教程、有趣案例...



# License

PaddleNLP遵循[Apache-2.0开源协议](https://github.com/PaddlePaddle/models/blob/develop/PaddleNLP/LICENSE)。