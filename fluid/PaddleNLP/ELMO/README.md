<h1 align="center">ELMO</h1>

## 介绍
ELMO(Embeddings from Language Models)是一种新型深度语境化词表征，可对词进行复杂特征(如句法和语义)和词在语言语境中的变化进行建模(即对多义词进行建模)。该模型支持多卡训练，训练速度比主流实现快约1倍,  验证在中文词法分析任务上f1值提升0.68%。

ELMO在大语料上以language model为训练目标，训练出bidirectional LSTM模型，利用LSTM产生词语的表征, 对下游NLP任务(如问答、分类、命名实体识别等）进行微调。

## 基本配置版本
Python 2.7

Paddlepaddle lastest版本

## 安装使用
下载代码后，把elmo训练任务跑起来，分为三步：

1.下载数据集和字典文件

利用download.sh下载词典和训练数据集和测试数据集。
```shell
cd data && sh download.sh
```

2.安装依赖库

```shell
sh download_thirdparty.sh
```

3.启动脚本训练模型
```shell
sh run.sh
```

## 预训练模型
预训练模型要点：
1.准备训练数据集trainset和测试数据集testset，准备一份词表如data/vocalubary.txt。
2.训练模型
3.测试模型结果
4.把checkpoint结果写入文件中。

## 数据预处理
以中文模型的预训练为例，可基于中文维基百科数据作为训练数据，先切成句子，再根据词典做分词。
```
本 书 介绍 了 中国 经济 发展 的 内外 平衡 问题 、 亚洲 金融 危机 十 周年 回顾 与 反思 、 实践 中 的 城乡 统筹 发展 、 未来 十 年 中国 需要 研究 的 重大 课题 、 科学 发展 与 新型 工业 化 等 方面 。
```
```
吴 敬 琏 曾经 提出 中国 股市 “ 赌场 论 ” ， 主张 维护 市场 规则 ， 保护 草根 阶层 生计 ， 被 誉 为 “ 中国 经济 学界 良心 ” ， 是 媒体 和 公众 眼中 的 学术 明星 
```

## 单机多卡训练
模型支持单机多卡训练，需要在run.sh里export CUDA_VISIBLE_DEVICES设置指定卡,如下所示：
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 如何利用ELMO做微调

1.下载ELMO Paddle官方发布Checkout文件

PaddlePaddle官方发布Checkout文件下载地址

2.在train部分中加载ELMO checkpoint文件
```shell
src_pretrain_model_path = '490001' #490001为ELMO checkpoint文件
def if_exist(var):
    path = os.path.join(src_pretrain_model_path, var.name)
    exist = os.path.exists(path)
    if exist:
        print('Load model: %s' % path)
    return exist

fluid.io.load_vars(executor=exe, dirname=src_pretrain_model_path, predicate=if_exist, main_program=main_program) 
```

3.在下游任务网络中加入bilm.py文件

4.在下游任务网络embedding部分加入ELMO网络的定义
```shell
#引入 bilm.py 文件
from bilm import elmo_encoder
from bilm import emb
#word为输入elmo部分切词后的字典
elmo_embedding = emb(word)
elmo_enc= elmo_encoder(elmo_embedding)

```

## 参考论文
《Deep contextualized word representations》 https://arxiv.org/abs/1802.05365
