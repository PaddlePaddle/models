
# 基于skip-gram的word2vector模型

## 介绍


## 运行环境
需要先安装PaddlePaddle Fluid

## 数据集
数据集使用的是来自1 Billion Word Language Model Benchmark的(http://www.statmt.org/lm-benchmark)的数据集.

下载数据集：
```bash
cd data && ./download.sh && cd ..
```

## 模型
本例子实现了一个skip-gram模式的word2vector模型。


## 数据准备
对数据进行预处理以生成一个词典。

```bash
python preprocess.py --data_path ./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled --dict_path data/1-billion_dict --is_local
```
如果您想使用我们支持的第三方词汇表，请将--other_dict_path设置为您存放将使用的词汇表的目录，并设置--with_other_dict使用它
如果您希望使用async executor来加速训练，需要先创建一个叫async_data的目录，然后使用以下命令：
```bash
python async_data_converter.py --train_data_path your_train_data_path --dict_path your_dict_path
```
如果您希望使用层次softmax则需要加上--with_hs，这个方法将会在您当前目录下刚刚创建的async_data目录下写入转换好用于async_executor的数据，如果您的数据集很大这个过程可能持续很久
## 训练
训练的命令行选项可以通过`python train.py -h`列出。

### 单机训练：

使用parallel executor
```bash
export CPU_NUM=1
python train.py \
        --train_data_path ./data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled \
        --dict_path data/1-billion_dict \
        --with_nce --is_local \
        2>&1 | tee train.log
```

使用async executor
```bash
python async_train.py --train_data_path ./async_data/ \
        --dict_path data/1-billion_dict --with_nce --with_hs \
        --epochs 1 --thread_num 1 --is_sparse --batch_size 100 --is_local 2>&1 | tee async_trainer1.log
```

如果您想使用我们支持的第三方词汇表，请将--other_dict_path设置为您存放将使用的词汇表的目录，并设置--with_other_dict使用它
### 分布式训练

本地启动一个2 trainer 2 pserver的分布式训练任务，分布式场景下训练数据会按照trainer的id进行切分，保证trainer之间的训练数据不会重叠，提高训练效率

```bash
sh cluster_train.sh
```

## 预测
在infer.py中我们在`build_test_case`方法中构造了一些test case来评估word embeding的效果：
我们输入test case（ 我们目前采用的是analogical-reasoning的任务：找到A - B = C - D的结构，为此我们计算A - B + D，通过cosine距离找最近的C，计算准确率要去除候选中出现A、B、D的候选 ）然后计算候选和整个embeding中所有词的余弦相似度，并且取topK（K由参数 --rank_num确定，默认为4）打印出来。

如：
对于：boy - girl + aunt = uncle  
0 nearest aunt:0.89
1 nearest uncle:0.70
2 nearest grandmother:0.67
3 nearest father:0.64

您也可以在`build_test_case`方法中模仿给出的例子增加自己的测试

要从测试文件运行测试用例，请将测试文件下载到“test”目录中
我们为每个案例提供以下结构的测试：
        `word1 word2 word3 word4`
所以我们可以将它构建成`word1  -  word2 + word3 = word4`

训练中预测：

```bash
python infer.py --infer_during_train 2>&1 | tee infer.log
```
使用某个model进行离线预测：

```bash
python infer.py --infer_once --model_output_dir ./models/[具体的models文件目录] 2>&1 | tee infer.log
```
## 在百度云上运行集群训练
1. 参考文档 [在百度云上启动Fluid分布式训练](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/user_guides/howto/training/train_on_baidu_cloud_cn.rst) 在百度云上部署一个CPU集群。
1. 用preprocess.py处理训练数据生成train.txt。
1. 将train.txt切分成集群机器份，放到每台机器上。
1. 用上面的 `分布式训练` 中的命令行启动分布式训练任务.
