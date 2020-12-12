# PaddleNLP transformer类预训练模型

随着深度学习的发展，NLP领域涌现了一大批高质量的transformer类预训练模型，多次刷新各种NLP任务SOTA。PaddleNLP为用户提供了常用的BERT、ERNIE等预训练模型，让用户能够方便快捷的使用各种transformer类模型，完成自己所需的任务。

## Transformer 类模型汇总

Model | Tokenizer | Supported Task| Pretrained Weight
---|---|---|---
 [BERT](https://arxiv.org/abs/1810.04805) | BertTokenizer|BertModel,<br> BertForQuestionAnswering,<br> BertForSequenceClassification,<br>BertForTokenClassification| `bert-base-uncased`,<br> `bert-large-uncased`, <br>`bert-base-multilingual-uncased`, <br>`bert-base-cased`,<br> `bert-base-chinese`,<br> `bert-base-multilingual-cased`,<br> `bert-large-cased`,<br> `bert-wwm-chinese`,<br> `bert-wwm-ext-chinese`
[ELECTRA](https://arxiv.org/abs/2003.10555) |ElectraTokenizer| ElectraModel,<br>ElectraForSequenceClassification,<br>ElectraForTokenClassification<br>|`electra-small`,<br> `electra-base`,<br> `electra-large`,<br> `chinese-electra-small`,<br> `chinese-electra-base`
[ERNIE](https://arxiv.org/abs/1904.09223)|ErnieTokenizer,<br>ErnieTinyTokenizer|ErnieModel,<br> ErnieForQuestionAnswering,<br> ErnieForSequenceClassification,<br> ErnieForTokenClassification| `ernie-1.0`,<br> `ernie-tiny`,<br> `ernie-2.0-en`,<br> `ernie-2.0-large-en`
[RoBERTa](https://arxiv.org/abs/1907.11692)|RobertaTokenizer| RobertaModel,<br>RobertaForQuestionAnswering,<br>RobertaForSequenceClassification,<br>RobertaForTokenClassification| `roberta-wwm-ext`,<br> `roberta-wwm-ext-large`,<br> `rbt3`,<br> `rbtl3`
[Transformer](https://arxiv.org/abs/1706.03762) |- | TransformerModel | -
