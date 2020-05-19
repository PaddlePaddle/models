# listwise

 以下是本例的简要目录结构及说明： 

```
├── README.md # 文档
├── evaluator.py # biRnn网络文件
├── utils.py # 通用函数
├── args.py # 参数脚本
├── train.py # 训练文件
├── infer.py # 预测文件
├── train_gpu.sh # gpu训练shell脚本
├── train_cpu.sh # cpu训练shell脚本
├── infer_gpu.sh # gpu预测shell脚本
├── infer_cpu.sh # cpu预测shell脚本
```

## 简介

[《Sequential Evaluation and Generation Framework for Combinatorial Recommender System》]( https://arxiv.org/pdf/1902.00245.pdf)是百度2019年发布的推荐系统融合模型，用于优化推荐序列的整体性能（如总点击），该模型由Generator和Evaluator两部分组成，Generator负责生成若干个候选序列，Evaluator负责从候选序列中筛选出最好的序列推荐给用户，达到最大化序列整体性能的目的。

本项目在paddlepaddle上实现该融合模型的Evaluator部分，构造数据集验证模型的正确性。

## 环境

 PaddlePaddle 1.7.0 

 python3.7 

## 单机训练

GPU环境

在train_gpu.sh脚本文件中设置好数据路径、参数。

```sh
CUDA_VISIBLE_DEVICES=0 python train.py --use_gpu 1\ #使用gpu
                                    --epochs 3\ 
                                    --batch_size 32\
                                    --model_dir './model_dir'\ #模型保存路径
                                    --embd_dim 16\  #embedding维度
                                    --hidden_size 128\ #biRNN隐层大小
                                    --item_vocab 200\ #item词典大小
                                    --user_vocab 200\ #user词典大小
                                    --item_len 5\ #序列长度
                                    --sample_size 100\ #构造数据集大小
                                    --base_lr 0.01 #学习率

```

修改脚本的可执行权限并运行

```
./train_gpu.sh
```

CPU环境

在train_cpu.sh脚本文件中设置好数据路径、参数。

```sh
python train.py --use_gpu 0\ #使用cpu
                --epochs 3\ 
                --batch_size 32\
                --model_dir './model_dir'\ #模型保存路径
                --embd_dim 16\  #embedding维度
                --hidden_size 128\ #biRNN隐层大小
                --item_vocab 200\ #item词典大小
                --user_vocab 200\ #user词典大小
                --item_len 5\ #序列长度
                --sample_size 100\ #构造数据集大小
                --base_lr 0.01 #学习率

```

修改脚本的可执行权限并运行

```sh
./train_cpu.sh
```

## 单机预测

GPU环境

在infer_gpu.sh脚本文件中设置好数据路径、参数。

```sh
CUDA_VISIBLE_DEVICES=0 python infer.py --use_gpu 1 \ #使用gpu
                                        --model_dir './model_dir'\
                                        --test_epoch 19 #选择哪一轮的模型参数

```

修改脚本的可执行权限并运行

```sh
./infer_gpu.sh
```

CPU环境

在infer_cpu.sh脚本文件中设置好数据路径、参数。

```sh
python infer.py --use_gpu 0\ #使用cpu
                --model_dir './model_dir'\
                --test_epoch 19 #选择哪一轮的模型参数

```

修改脚本的可执行权限并运行

```
./infer_cpu.sh
```

## 模型效果

在测试集的效果如下：

```
W0518 21:38:58.030905  8105 device_context.cc:237] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 9.2, Runtime API Version: 9.0
W0518 21:38:58.035158  8105 device_context.cc:245] device: 0, cuDNN Version: 7.3.
2020-05-18 21:38:59,553-INFO: epoch_id: 0, batch_time: 0.01643s, loss: 0.69452, auc: 0.47282
2020-05-18 21:38:59,567-INFO: epoch_id: 0, batch_time: 0.01314s, loss: 0.77172, auc: 0.49025
2020-05-18 21:38:59,580-INFO: epoch_id: 0, batch_time: 0.01261s, loss: 0.69282, auc: 0.51839
......
2020-05-18 21:39:03,702-INFO: epoch_id: 2, batch_time: 0.01287s, loss: 0.69431, auc: 0.50265
2020-05-18 21:39:03,715-INFO: epoch_id: 2, batch_time: 0.01278s, loss: 0.69272, auc: 0.50267
2020-05-18 21:39:03,728-INFO: epoch_id: 2, batch_time: 0.01274s, loss: 0.69340, auc: 0.50267
```

