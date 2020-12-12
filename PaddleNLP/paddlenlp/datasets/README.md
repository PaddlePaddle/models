# Paddlenlp Datasets

| 任务类型   | 数据集名称   | 简介 | 调用方法 |
| ------ | ----  | --------- | ------ | ---- |
| 阅读理解| SQaAD | 斯坦福问答数据集，包括squad1.1和squad2.0|paddlenlp.datasets.SQuAD |
| 阅读理解| DuReader-yesno | 千言数据集：阅读理解，判断答案极性|paddlenlp.datasets.DuReaderYesNo |
| 阅读理解| DuReader-robust | 千言数据集：阅读理解，答案原文抽取|paddlenlp.datasets.DuReaderRobust |
| 序列标注| lexical_analysis_dataset | 词法分析数据集|[lexical_analysis](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/examples/lexical_analysis) |
| 序列标注| Conll05 | 语义角色标注数据集| paddle.text.datasets.Conll05st|
| 序列标注| MSRA_NER | MSRA 命名实体识别数据集| paddlenlp.datasets.MSRA_NER|
| 机器翻译| IWSLT15 | IWSLT'15 English-Vietnamese data 英语-越南语翻译数据集| paddlenlp.datasets.IWSLT15|
| 机器翻译| WMT | WMT EN-DE 英语-德语翻译数据集| paddlenlp.datasets|
| 文本分类| CoLA | 单句分类任务，二分类，判断句子是否合法| paddlenlp.datasets.GlueCoLA|
| 文本分类| SST-2 | 单句分类任务，二分类，判断句子情感极性| paddlenlp.datasets.GlueSST2|
| 文本分类| MRPC | 句对匹配任务，二分类，判断句子对是否是相同意思| paddlenlp.datasets.GlueMRPC|
| 文本分类| STSB | 计算句子对相似性，分数为1~5| paddlenlp.datasets.GlueSTSB|
| 文本分类| QQP | 判定句子对是否等效，等效、不等效两种情况，二分类任务| paddlenlp.datasets.GlueQQP|
| 文本分类| MNLI | 句子对，一个前提，一个是假设。前提和假设的关系有三种情况：蕴含（entailment），矛盾（contradiction），中立（neutral）。句子对三分类问题| paddlenlp.datasets.GlueMNLI|
| 文本分类| QNLI | 判断问题（question）和句子（sentence）是否蕴含，蕴含和不蕴含，二分类| paddlenlp.datasets.GlueQNLI|
| 文本分类| RTE | 判断句对是否蕴含，句子1和句子2是否互为蕴含，二分类任务| paddlenlp.datasets.GlueRTE|
| 文本分类| WNLI | 判断句子对是否相关，相关或不相关，二分类任务| paddlenlp.datasets.GlueWNLI|
| 文本分类| LCQMC | A Large-scale Chinese Question Matching Corpus 语义匹配数据集| paddlenlp.datasets.LCQMC|
| 文本分类| ChnSentiCorp | 中文评论情感分析语料| paddlenlp.datasets.ChnSentiCorp|
| 文本分类| IMDB | IMDB电影评论情感分析数据集| paddle.text.datasets.Imdb|
| 文本分类| Movielens | Movielens 1-M电影评级数据集| paddle.text.datasets.Movielens|
| 语料库| yahoo | 雅虎中文语料库| [VAE](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/examples/text_generation/vae-seq2seq)|
| 语料库| PTB | Penn Treebank Dataset 语料库| paddlenlp.datasets.PTB|
| 语料库| 1 Billon words | 1 Billion Word Language Model Benchmark R13 Output 基准语料库| [ELMo](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/examples/language_model/elmo)|
| 时序预测| CSSE COVID-19 |约翰·霍普金斯大学系统科学与工程中心新冠病例数据 | [time_series](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/examples/time_series)|
| 时序预测| UCIHoussing | 波士顿房价预测数据集 | paddle.text.datasets.UCIHousing|
