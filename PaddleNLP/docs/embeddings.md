# Embedding 模型汇总

PaddleNLP提供多个开源的预训练Embedding模型，用户仅需在使用`paddlenlp.embeddings.TokenEmbedding`时，指定预训练模型的名称，即可加载相对应的预训练模型。以下为PaddleNLP所支持的预训练Embedding模型，其名称用作`paddlenlp.embeddings.TokenEmbedding`的参数。命名方式为：\${模型}.\${语料}.\${co-occurrence type}.dim\${维度}。

## 中文词向量

根据不同的上下文特征为每个语料训练多个中心词向量。
以下预训练模型由[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)提供。


| 语料 | Word | Word + Ngram | Word + Character | Word + Character + Ngram |
| -------------------- | ----   | ---- | ----   | ---- |
| Baidu Encyclopedia 百度百科                   | w2v.baidu_encyclopedia.target.word-word.dim300 | w2v.baidu_encyclopedia.target.word-ngram.1-2.dim300 | w2v.baidu_encyclopedia.target.word-character.char1-2.dim300 | w2v.baidu_encyclopedia.target.bigram-char.dim300 |
| Wikipedia_zh 中文维基百科	                     | w2v.wiki.target.word-word.dim300 | w2v.wiki.target.word-bigram.dim300 | w2v.wiki.target.word-char.dim300 | w2v.wiki.target.bigram-char.dim300 |
| People's Daily News 人民日报                  | w2v.people_daily.target.word-word.dim300 | w2v.people_daily.target.word-bigram.dim300 | w2v.people_daily.target.word-char.dim300 | w2v.people_daily.target.bigram-char.dim300 |
| Sogou News 搜狗新闻	                        | w2v.sogou.target.word-word.dim300 | w2v.sogou.target.word-bigram.dim300 | w2v.sogou.target.word-char.dim300 | w2v.sogou.target.bigram-char.dim300 |
| Financial News 金融新闻	                    | w2v.financial.target.word-word.dim300 | w2v.financial.target.word-bigram.dim300 | w2v.financial.target.word-char.dim300 | w2v.financial.target.bigram-char.dim300 |
| Zhihu_QA 知乎问答		                        | w2v.zhihu.target.word-word.dim300 | w2v.zhihu.target.word-bigram.dim300 | w2v.zhihu.target.word-char.dim300 | w2v.zhihu.target.bigram-char.dim300 |
| Weibo 微博		                           | w2v.weibo.target.word-word.dim300 | w2v.weibo.target.word-bigram.dim300 | w2v.weibo.target.word-char.dim300 | w2v.weibo.target.bigram-char.dim300 |
| Literature 文学作品		                    | w2v.literature.target.word-word.dim300 | w2v.literature.target.word-bigram.dim300 | w2v.literature.target.word-char.dim300 | w2v.literature.target.bigram-char.dim300 |
| Complete Library in Four Sections 四库全书	| w2v.sikuquanshu.target.word-word.dim300 | w2v.sikuquanshu.target.word-bigram.dim300 | 无 | 无 |
| Mixed-large 综合                             | 暂无 | 暂无 | 暂无 | 暂无 |

特别地，对于百度百科语料，在不同的 Co-occurrence类型下分别提供了中心词与背景词向量：

| Co-occurrence 类型           | 中心词向量 | 背景词向量  |
| --------------------------- | ------   | ---- |
|    Word → Word              | w2v.baidu_encyclopedia.target.word-word.dim300     |   w2v.baidu_encyclopedia.context.word-word.dim300    |
|    Word → Ngram (1-2)	      |  w2v.baidu_encyclopedia.target.word-ngram.1-2.dim300    |   暂无    |
|    Word → Ngram (1-3)       |  暂无    |   暂无    |
|    Ngram (1-2) → Ngram (1-2)|  暂无   |   暂无    |
|    Word → Character (1)	  |  w2v.baidu_encyclopedia.target.word-character.char1-1.dim300    |  w2v.baidu_encyclopedia.context.word-character.char1-1.dim300     |
|    Word → Character (1-2)   |  w2v.baidu_encyclopedia.target.word-character.char1-2.dim300    |  w2v.baidu_encyclopedia.context.word-character.char1-2.dim300     |
|    Word → Character (1-4)   |  w2v.baidu_encyclopedia.target.word-character.char1-4.dim300    |  w2v.baidu_encyclopedia.context.word-character.char1-4.dim300     |
|    Word → Word (left/right) |   暂无   |   暂无    |
|    Word → Word (distance)   |   暂无   |   暂无    |

## 英文词向量

待更新。
