# youtube dnn

 以下是本例的简要目录结构及说明：

```
├── README.md # 文档
├── youtubednn.py # youtubednn.py网络文件
├── args.py # 参数脚本
├── train.py # 训练文件
├── infer.py # 预测文件
├── train_gpu.sh # gpu训练shell脚本
├── train_cpu.sh # cpu训练shell脚本
├── infer_gpu.sh # gpu预测shell脚本
├── infer_cpu.sh # cpu预测shell脚本
├── get_topk.py # 获取user最有可能点击的k个video
├── rec_topk.sh # 推荐shell脚本
```

## 简介

[《Deep Neural Networks for YouTube Recommendations》](https://link.zhihu.com/?target=https%3A//static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf) 这篇论文是google的YouTube团队在推荐系统上DNN方面的尝试，是经典的向量化召回模型，主要通过模型来学习用户和物品的兴趣向量，并通过内积来计算用户和物品之间的相似性，从而得到最终的候选集。YouTube采取了两层深度网络完成整个推荐过程：

1.第一层是**Candidate Generation Model**完成候选视频的快速筛选，这一步候选视频集合由百万降低到了百的量级。

2.第二层是用**Ranking Model**完成几百个候选视频的精排。

本项目在paddlepaddle上完成YouTube dnn的召回部分Candidate Generation Model，分别获得用户和物品的向量表示，从而后续可以通过其他方法（如用户和物品的余弦相似度）给用户推荐物品。

由于原论文没有开源数据集，本项目随机构造数据验证网络的正确性。

## 环境

 PaddlePaddle 1.7.0 

 python3.7 

## 单机训练

GPU环境

在train_gpu.sh脚本文件中设置好数据路径、参数。

```sh
CUDA_VISIBLE_DEVICES=0 python train.py --use_gpu 1\ #使用gpu
                                    --batch_size 32\
                                    --epochs 20\
                                    --watch_vec_size 64\ #特征维度
                                    --search_vec_size 64\
                                    --other_feat_size 64\
                                    --output_size 100\ 
                                    --model_dir 'model_dir'\ #模型保存路径
                                    --test_epoch 19\
                                    --base_lr 0.01\
                                    --video_vec_path './video_vec.csv' #得到物品向量文件路径
```

执行脚本

```sh
sh train_gpu.sh
```

CPU环境

在train_cpu.sh脚本文件中设置好数据路径、参数。

```sh
python train.py --use_gpu 0\ #使用cpu
                --batch_size 32\
                --epochs 20\
                --watch_vec_size 64\ #特征维度
                --search_vec_size 64\
                --other_feat_size 64\
                --output_size 100\ 
                --model_dir 'model_dir'\ #模型保存路径
                --test_epoch 19\
                --base_lr 0.01\
                --video_vec_path './video_vec.csv' #得到物品向量文件路径
```

执行脚本

```
sh train_cpu.sh
```

## 单机预测

GPU环境

在infer_gpu.sh脚本文件中设置好数据路径、参数。

```sh
CUDA_VISIBLE_DEVICES=0 python infer.py --use_gpu 1 \ #使用gpu
                                    --test_epoch 19 \ #采用哪一轮模型来预测
                                    --model_dir './model_dir' \ #模型路径
                                    --user_vec_path './user_vec.csv' #用户向量路径
```

执行脚本

```sh
sh infer_gpu.sh
```

CPU环境

在infer_cpu.sh脚本文件中设置好数据路径、参数。

```sh
python infer.py --use_gpu 0 \ #使用cpu
                --test_epoch 19 \ #采用哪一轮模型来预测
                --model_dir './model_dir' \ #模型路径
                --user_vec_path './user_vec.csv' #用户向量路径
```

执行脚本

```sh
sh infer_cpu.sh
```

## 模型效果

构造数据集进行训练：

```
W0512 23:12:36.044643  2124 device_context.cc:237] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.0
W0512 23:12:36.050058  2124 device_context.cc:245] device: 0, cuDNN Version: 7.3.
2020-05-12 23:12:37,681-INFO: epoch_id: 0, batch_time: 0.00719s, loss: 4.68754, acc: 0.00000
2020-05-12 23:12:37,686-INFO: epoch_id: 0, batch_time: 0.00503s, loss: 4.54141, acc: 0.03125
2020-05-12 23:12:37,691-INFO: epoch_id: 0, batch_time: 0.00419s, loss: 4.92227, acc: 0.00000
```

通过计算每个用户和每个物品的余弦相似度，给每个用户推荐topk视频：

```
user:0, top K videos:[93, 73, 6, 20, 84]
user:1, top K videos:[58, 0, 46, 86, 71]
user:2, top K videos:[52, 51, 47, 82, 19]
......
user:96, top K videos:[0, 52, 86, 45, 11]
user:97, top K videos:[0, 52, 45, 58, 28]
user:98, top K videos:[58, 24, 49, 36, 46]
user:99, top K videos:[0, 47, 44, 72, 51]
```

