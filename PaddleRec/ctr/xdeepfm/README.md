
# xDeepFM for CTR Prediction

## 简介
使用PaddlePaddle复现论文 "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems" 。论文[开源代码](https://github.com/Leavingseason/xDeepFM) 。

## 数据集

demo数据集，在data目录下执行命令，下载数据
```bash
sh download.sh
```

## 环境
- PaddlePaddle 1.5

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
```bash
sh cluster_train.sh
```

预测
```bash
python infer.py --model_output_dir cluster_model --test_epoch 10 --use_gpu=0
```
注意:

- 本地模拟需要关闭代理

- 0号trainer保存模型参数

- 每次训练完成后需要手动停止pserver进程，使用以下命令查看pserver进程：
  >ps -ef | grep python

- 数据读取使用dataset模式，目前仅支持运行在Linux环境下
