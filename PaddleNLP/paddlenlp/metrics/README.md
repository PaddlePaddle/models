# paddlenlp.metrics

目前paddlenlp提供以下评价指标：

| Metric                                                     | 简介                                                         |
| ---------------------------------------------------------- | :----------------------------------------------------------- |
| Perplexity                                                 | 困惑度，常用来衡量语言模型优劣，也可用于机器翻译、文本生成等任务。 |
| BLEU(bilingual evaluation understudy)                      | 机器翻译常用评价指标                                         |
| Rouge-L(Recall-Oriented Understudy for Gisting Evaluation) | 评估自动文摘以及机器翻译的指标                               |
| AccuracyAndF1                                              | 准确率及F1，可用于glue中的mrpc and qqp任务                   |
| PearsonAndSpearman                                         | 皮尔森相关性系数和斯皮尔曼相关系数。可用于glue中的sts-b任务  |
| Mcc                                                        | 马修斯相关系数，用以测量二分类的分类性能的指标。可用于glue中的cola任务 |
| ChunkEvaluator                                             | 计算了块检测的精确率、召回率和F1-score。常用于序列标记任务，如命名实体识别（NER） |
| Squad                                                      | 用于SQuAD的评价指标                                          |
| Dureader                                                   | 用于Dureader的评价指标                                       |
