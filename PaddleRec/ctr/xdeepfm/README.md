
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

## 结果
训练集训练10轮后，测试集的LogLoss : `0.48657` 和 AUC : `0.7308`。
