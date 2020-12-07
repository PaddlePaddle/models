# paddlenlp.embeddings

## Embedding快速复用热启

初定三个模型的Embedding数据，SimNet，word2vec，FastText

使用LAC切词+大规模中文语料快速训练多个中文的embedding，注意筛选高质量词表

* SimNet 大搜数据中文
* word2vec 中英文
* fasttext 中英文

## 再提供Fleet的word2vec训练入口