# Word Embedding with PaddleNLP

## 简介

PaddleNLP已预置多个公开的预训练Embedding，用户可以通过使用`paddle.embeddings.TokenEmbedding`接口加载预训练Embedding，从而提升训练效果。以下通过文本分类训练的例子展示`paddle.embeddings.TokenEmbedding`对训练提升的效果。


## 快速开始

### 安装说明

* PaddlePaddle 安装

   本项目依赖于 PaddlePaddle 2.0 及以上版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

* PaddleNLP 安装

   ```shell
   pip install paddlenlp
   ```

* 环境依赖

   本项目依赖于jieba分词，请在运行本项目之前，安装jieba，如`pip install -U jieba`

   Python的版本要求 3.6+，其它环境请参考 PaddlePaddle [安装说明](https://www.paddlepaddle.org.cn/install/quick/zh/2.0rc-linux-docker) 部分的内容

### 下载词表

下载词汇表文件dict.txt，用于构造词-id映射关系。

```bash
wget https://paddlenlp.bj.bcebos.com/data/dict.txt
```

### 启动训练

我们以中文情感分类公开数据集ChnSentiCorp为示例数据集，可以运行下面的命令，在训练集（train.tsv）上进行模型训练，并在开发集（dev.tsv）验证。实验输出的日志保存在use_token_embedding.txt和use_normal_embedding.txt。使用PaddlePaddle框架的Embedding在ChnSentiCorp下非常容易过拟合，因此调低了它的学习率。

CPU 启动：

```
nohup python train.py --vocab_path='./dict.txt' --use_gpu=False --lr=5e-4 --batch_size=64 --epochs=20 --use_token_embedding=True --vdl_dir='./vdl_dir' >use_token_embedding.txt 2>&1 &

nohup python train.py --vocab_path='./dict.txt' --use_gpu=False --lr=1e-4 --batch_size=64 --epochs=20 --use_token_embedding=False --vdl_dir='./vdl_dir'>use_normal_embedding.txt 2>&1 &
```

GPU 启动：
```
export CUDA_VISIBLE_DEVICES=0

nohup python train.py --vocab_path='./dict.txt' --use_gpu=True --lr=5e-4 --batch_size=64 --epochs=20 --use_token_embedding=True --vdl_dir='./vdl_dir' > use_token_embedding.txt 2>&1 &

# 如显存不足，可以先等第一个训练完成再启动该训练
nohup python train.py --vocab_path='./dict.txt' --use_gpu=True --lr=1e-4 --batch_size=64 --epochs=20 --use_token_embedding=False --vdl_dir='./vdl_dir' > use_normal_embedding.txt 2>&1 &
```

以上参数表示：

* `vocab_path`: 词汇表文件路径。
* `use_gpu`: 是否使用GPU进行训练， 默认为`False`。
* `lr`: 学习率， 默认为5e-4。
* `batch_size`: 运行一个batch大小，默认为64。
* `epochs`: 训练轮次，默认为5。
* `use_token_embedding`: 是否使用PaddleNLP的TokenEmbedding，默认为True。
* `vdl_dir`: VisualDL日志目录。训练过程中的VisualDL信息会在该目录下保存。默认为`./vdl_dir`

该脚本还提供以下参数：

* `save_dir`: 模型保存目录。
* `init_from_ckpt`: 恢复模型训练的断点路径。
* `embedding_name`: 预训练Embedding名称，默认为`w2v.baidu_encyclopedia.target.word-word.dim300`. 支持的预训练Embedding可参考[Embedding 模型汇总](../../docs/embeddings.md)。

### 启动VisualDL

推荐使用VisualDL查看实验对比。以下为VisualDL的启动命令，其中logdir参数指定的目录需要与启动训练时指定的`vdl_dir`相同。（更多VisualDL的用法，可参考[VisualDL使用指南](https://github.com/PaddlePaddle/VisualDL#2-launch-panel)）

```
nohup visualdl --logdir ./vdl_dir --port 8888 --host 0.0.0.0 &
```

### 训练效果对比

在Chrome浏览器输入 `ip:8888` (ip为启动VisualDL机器的IP)。

以下为示例实验效果对比图，蓝色是使用`paddle.embeddings.TokenEmbedding`进行的实验，绿色是使用没有加载预训练模型的Embedding进行的实验。可以看到，使用`paddle.embeddings.TokenEmbedding`的训练，其验证acc变化趋势上升，并收敛于0.90左右，收敛后相对平稳，不容易过拟合。而没有使用`paddle.embeddings.TokenEmbedding`的训练，其验证acc变化趋势向下，并收敛于0.86左右。从示例实验可以观察到，使用`paddle.embedding.TokenEmbedding`能提升训练效果。

Eval Acc：

![eval acc](https://user-images.githubusercontent.com/16698950/102076935-79ac5480-3e43-11eb-81f8-6e509c394fbf.png)

|                                     |    Best Acc    |
| ------------------------------------| -------------  |
| paddle.nn.Embedding                 |    0.8965      |
| paddelnlp.embeddings.TokenEmbedding |    0.9082      |

## 致谢
- 感谢 [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)提供Word2Vec中文Embedding来源。
