# DSSM

```
├── README.md			 # 文档
├── dssm.py				 # dssm网络结构
├── args.py				 # 参数脚本
├── infer.py			 # 预测脚本
├── train_gpu.sh		 # gpu训练shell脚本
├── train_cpu.sh		 # cpu训练shell脚本
├── infer_gpu.sh		 # gpu预测shell脚本
├── infer_cpu.sh		 # cpu预测shell脚本
```

## 简介

DSSM[《Learning Deep Structured Semantic Models for Web Search using Clickthrough Data》](  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf  )即基于深度网络的语义模型，其核心思想是将query和doc映射到共同维度的语义空间中，通过最大化query和doc语义向量之间的余弦相似度，从而训练得到隐含语义模型，达到检索的目的，并通过word hashing方法来减少输入向量的维度。DSSM有很广泛的应用，比如：搜索引擎检索，广告相关性，问答系统，机器翻译等。

本项目按照论文的网络结构在paddlepaddle上实现DSSM模型，并构造数据集验证网络的正确性。

## 环境

 PaddlePaddle 1.7.0 

 python3.7 

## 数据集说明

由于论文没有公开数据集，本项目构造数据验证网络的正确性，其说明如下：

query：随机构造的query向量表示

doc_pos：随机构造doc正例向量表示

doc_neg_0~3为四个doc负例向量表示

## 单机训练

GPU环境

在train_gpu.sh脚本文件中设置好参数。

```sh
CUDA_VISIBLE_DEVICES=0 python dssm.py --use_gpu 1 \ #使用gpu
                                   --batch_size 16 \ #batch大小
                                   --TRIGRAM_D 1000 \ #输入向量维度
                                   --L1_N 300 \ #第一层mlp大小
                                   --L2_N 300 \ #第二层mlp大小
                                   --L3_N 128 \ #第三层mlp大小
                                   --Neg 4 \ #负采样个数
                                   --base_lr 0.01 \ #学习率
                                   --model_dir 'model_dir' #模型保存路径
```

修改脚本的可执行权限并运行

```shell
./train_gpu.sh
```

CPU环境

在train_cpu.sh脚本文件中设置好参数。

```sh
python dssm.py --use_gpu 0 \ #使用cpu
                --batch_size 16 \ #batch大小
                --TRIGRAM_D 1000 \ #输入向量维度
                --L1_N 300 \ #第一层mlp大小
                --L2_N 300 \ #第二层mlp大小
                --L3_N 128 \ #第三层mlp大小
                --Neg 4 \ #负采样个数
                --base_lr 0.01 \ #学习率
                --model_dir 'model_dir' #模型保存路径
```

修改脚本的可执行权限并运行

```
./train_cpu.sh
```

# 预测

GPU环境

在infer_gpu.sh脚本文件中设置好参数。

```sh
CUDA_VISIBLE_DEVICES=0 python infer.py --use_gpu 1 \ #使用gpu
                                        --model_dir 'model_dir' #模型路径
```

修改脚本的可执行权限并运行

```sh
./infer_gpu.sh
```

CPU环境

在infer_cpu.sh脚本文件中设置好参数。

```sh
python infer.py --use_gpu 0 \ #使用cpu
               --model_dir 'model_dir' #模型路径
```

修改脚本的可执行权限并运行

```
./infer_cpu.sh
```



## 模型效果

随机构造4个负样本进行训练，可见loss达到收敛状态。

```txt
W0422 15:36:37.033936  1627 device_context.cc:237] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 9.2, Runtime API Version: 9.0
W0422 15:36:37.039381  1627 device_context.cc:245] device: 0, cuDNN Version: 7.3.
2020-04-22 15:36:38,718-INFO: epoch_id: 0, batch_time: 0.02135s, loss: 25.05417
2020-04-22 15:36:38,734-INFO: epoch_id: 1, batch_time: 0.01645s, loss: 16.14477
2020-04-22 15:36:38,750-INFO: epoch_id: 2, batch_time: 0.01573s, loss: 12.89269
2020-04-22 15:36:38,766-INFO: epoch_id: 3, batch_time: 0.01551s, loss: 11.51237
2020-04-22 15:36:38,785-INFO: epoch_id: 4, batch_time: 0.01890s, loss: 10.70215
......

2020-04-22 15:36:40,267-INFO: epoch_id: 95, batch_time: 0.01512s, loss: 7.13324
2020-04-22 15:36:40,282-INFO: epoch_id: 96, batch_time: 0.01502s, loss: 7.14063
2020-04-22 15:36:40,298-INFO: epoch_id: 97, batch_time: 0.01506s, loss: 7.13577
2020-04-22 15:36:40,314-INFO: epoch_id: 98, batch_time: 0.01512s, loss: 7.13683
2020-04-22 15:36:40,329-INFO: epoch_id: 99, batch_time: 0.01519s, loss: 7.13883
```

预测阶段可算出query和doc的相似度

```txt
W0422 15:40:16.847975  1752 device_context.cc:237] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 9.2, Runtime API Version: 9.0
W0422 15:40:16.853554  1752 device_context.cc:245] device: 0, cuDNN Version: 7.3.
2020-04-22 15:40:18,589-INFO: query_doc_sim: 0.99267
2020-04-22 15:40:18,593-INFO: query_doc_sim: 0.99123
2020-04-22 15:40:18,596-INFO: query_doc_sim: 0.99198
2020-04-22 15:40:18,599-INFO: query_doc_sim: 0.99010
2020-04-22 15:40:18,602-INFO: query_doc_sim: 0.98832
......
2020-04-22 15:40:18,854-INFO: query_doc_sim: 0.99079
2020-04-22 15:40:18,857-INFO: query_doc_sim: 0.98585
2020-04-22 15:40:18,860-INFO: query_doc_sim: 0.98702
2020-04-22 15:40:18,863-INFO: query_doc_sim: 0.99151
2020-04-22 15:40:18,867-INFO: query_doc_sim: 0.98917
```

