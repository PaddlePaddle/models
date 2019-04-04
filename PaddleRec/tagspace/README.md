# TagSpace

以下是本例的简要目录结构及说明：

```text
.
├── README.md            # 文档
├── train.py             # 训练脚本
├── infer.py             # 预测脚本
├── net.py               # 网络结构
├── text2paddle.py       # 文本数据转paddle数据
├── cluster_train.py     # 多机训练
├── cluster_train.sh     # 多机训练脚本
├── utils                # 通用函数
├── vocab_text.txt       # 小样本文本字典
├── vocab_tag.txt        # 小样本类别字典
├── train_data           # 小样本训练目录
└── test_data            # 小样本测试目录

```


## 简介

TagSpace模型的介绍可以参阅论文[#TagSpace: Semantic Embeddings from Hashtags](https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags/)。

Tagspace模型学习文本及标签的embedding表示，应用于工业级的标签推荐，具体应用场景有feed新闻标签推荐。


## 数据下载及预处理

数据地址： [ag news dataset](https://github.com/mhjabreel/CharCNN/tree/master/data/)

备份数据地址：[ag news dataset](https://paddle-tagspace.bj.bcebos.com/data.tar)

数据格式如下

```
"3","Wall St. Bears Claw Back Into the Black (Reuters)","Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again."
```

备份数据解压后，将文本数据转为paddle数据，先将数据放到训练数据目录和测试数据目录
```
mv train.csv raw_big_train_data
mv test.csv raw_big_test_data
```

运行脚本text2paddle.py 生成paddle输入格式
```
python text2paddle.py raw_big_train_data/ raw_big_test_data/ train_big_data test_big_data big_vocab_text.txt big_vocab_tag.txt
```

## 单机训练
'--use_cuda 1' 表示使用gpu, 0表示使用cpu, '--parallel 1' 表示使用多卡

小数据训练（样例中的数据已经准备，可跳过上一节的数据准备，直接运行命令）

GPU 环境
```
CUDA_VISIBLE_DEVICES=0 python train.py  --use_cuda 1
```
CPU 环境
```
python train.py
```

全量数据单机单卡训练
```
CUDA_VISIBLE_DEVICES=0 python train.py --use_cuda 1 --train_dir train_big_data/ --vocab_text_path big_vocab_text.txt --vocab_tag_path big_vocab_tag.txt --model_dir big_model --batch_size 500
```
全量数据单机多卡训练

```
python train.py --train_dir train_big_data/ --vocab_text_path big_vocab_text.txt --vocab_tag_path big_vocab_tag.txt --model_dir big_model --batch_size 500 --parallel 1
```

## 预测
小数据预测
```
python infer.py
```

全量数据预测
```
python infer.py --model_dir big_model --vocab_tag_path big_vocab_tag.txt --test_dir test_big_data/
```

## 本地模拟多机
运行命令
```
sh cluster_train.py
```
