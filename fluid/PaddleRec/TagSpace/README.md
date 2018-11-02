# TagSpace

以下是本例的简要目录结构及说明：

```text
.
├── README.md            # 文档
├── train.py             # 训练脚本
├── utils                # 通用函数
├── small_train.txt      # 小样本训练集
└── small_test.txt       # 小样本测试集

```


## 简介

TagSpace模型的介绍可以参阅论文[#TagSpace: Semantic Embeddings from Hashtags](https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags/)，在本例中，我们实现了TagSpace的模型。
## 数据下载

[ag news dataset](https://github.com/mhjabreel/CharCNN/tree/master/data/ag_news_csv)

数据格式如下

```
"3","Wall St. Bears Claw Back Into the Black (Reuters)","Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again."
```


## 训练
'--use_cuda 1' 表示使用gpu, 缺省表示使用cpu 

GPU 环境
运行命令 `CUDA_VISIBLE_DEVICES=0 python train.py train_file test_file --use_cuda 1` 开始训练模型。
```
CUDA_VISIBLE_DEVICES=0 python train.py small_train.txt small_test.txt --use_cuda 1
```
CPU 环境
运行命令 `python train.py train_file test_file` 开始训练模型。
```
python train.py small_train.txt small_test.txt
```

## 未来工作

添加预测部分

添加多种负例采样方式


