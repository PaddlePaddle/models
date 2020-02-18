
# xDeepFM for CTR Prediction

## 简介
使用PaddlePaddle复现论文 "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems" 。论文[开源代码](https://github.com/Leavingseason/xDeepFM) 。

## 数据集

demo数据集，在data目录下执行命令，下载数据
```bash
python download.py
```

## 环境
- **要求使用PaddlePaddle 1.6及以上版本或适当的develop版本。**

## 单机训练
```bash
python local_train.py --model_output_dir models
```
训练过程中每隔固定的steps（默认为50）输出当前loss和auc，可以在args.py中调整print_steps。

## 单机预测

```bash
python infer.py --model_output_dir models --test_epoch 10
```
test_epoch设置加载第10轮训练的模型。

注意：最后的 log info是测试数据集的整体 Logloss 和 AUC。

## 单机结果
训练集训练10轮后，测试集的LogLoss : `0.48657` 和 AUC : `0.7308`。

## 多机训练
运行命令本地模拟多机场景，默认使用2 X 2模式，即2个pserver，2个trainer的方式组网训练。

**注意：在多机训练中，建议使用Paddle 1.6版本以上或[最新版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/Tables.html#whl-dev)。**

数据下载同上面命令。
```bash
# 该sh不支持Windows
sh cluster_train.sh
```
参数说明：
- train_data_dir: 训练数据目录
- model_output_dir: 模型保存目录
- is_local: 是否单机本地训练(单机模拟多机分布式训练是为0)
- is_sparse: embedding是否使用sparse。如果没有设置，默认是False
- role: 进程角色(pserver或trainer)
- endpoints: 所有pserver地址和端口
- current_endpoint: 当前pserver(role是pserver)端口和地址
- trainers: trainer数量

其他参数见cluster_train.py

预测
```bash
python infer.py --model_output_dir cluster_model --test_epoch 10 --use_gpu=0
```
注意:

- 本地模拟需要关闭代理，e.g. unset http_proxy, unset https_proxy

- 0号trainer保存模型参数

- 每次训练完成后需要手动停止pserver进程，使用以下命令查看pserver进程：

>ps -ef | grep python

- 数据读取使用dataset模式，目前仅支持运行在Linux环境下
