# 基于skip-gram的word2vector模型

以下是本例的简要目录结构及说明：

```text
.
├── infer.py            # 预测脚本
├── net.py              # 网络结构
├── preprocess.py       # 预处理脚本，包括构建词典和预处理文本
├── reader.py           # 训练阶段的文本读写
├── README.md           # 使用说明
├── train.py            # 训练函数
└── utils.py            # 通用函数

```

## 介绍
本例实现了skip-gram模式的word2vector模型。


## 数据集
大数据集使用的是来自1 Billion Word Language Model Benchmark的(http://www.statmt.org/lm-benchmark)的数据集.下载命令如下

```bash
wget https://paddle-zwh.bj.bcebos.com/1-billion-word-language-modeling-benchmark-r13output.tar
```

小数据集使用1700w个词的text8数据集，下载命令如下

下载数据集：
```bash
wget https://paddle-zwh.bj.bcebos.com/text.tar
```


## 数据预处理
以下以小数据为例进行预处理。

大数据集注意解压后以training-monolingual.tokenized.shuffled 目录为预处理目录，和小数据集的text目录并列。

根据英文语料生成词典, 中文语料可以通过修改text_strip

```bash
python preprocess.py --build_dict --build_dict_corpus_dir data/text/ --dict_path data/test_build_dict
```

根据词典将文本转成id, 同时进行downsample，按照概率过滤常见词。

```bash
python preprocess.py --filter_corpus --dict_path data/test_build_dict --input_corpus_dir data/text/ --output_corpus_dir data/convert_text8 --min_count 5 --downsample 0.001
```

## 训练
cpu 单机多线程训练

```bash
OPENBLAS_NUM_THREADS=1 CPU_NUM=5 python train.py --train_data_dir data/convert_text8 --dict_path data/test_build_dict --num_passes 10 --batch_size 100 --model_output_dir v1_cpu5_b100_lr1dir --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse
```

## 预测
测试集下载命令如下

```bash
#大数据集测试集
wget https://paddle-zwh.bj.bcebos.com/test_dir.tar
#小数据集测试集
wget https://paddle-zwh.bj.bcebos.com/test_mid_dir.tar
```

预测命令
```bash
python infer.py --infer_epoch --test_dir data/test_mid_dir/ --dict_path data/test_build_dict_word_to_id_ --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0
```
