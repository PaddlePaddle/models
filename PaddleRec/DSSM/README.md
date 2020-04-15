# DSSM

```
├── README.md			 # 文档
├── net.py				 # ESMM网络结构
├── train.py			 # ESMM模型训练脚本
├── args.py				 # 参数脚本
├── infer.py			 # ESMM模型预测脚本
├── gpu_train.sh		 # gpu训练shell脚本
├── cpu_train.sh		 # cpu训练shell脚本
├── cpu_infer.sh		 # cpu预测shell脚本
```

## 简介

DSSM[《Learning Deep Structured Semantic Models for Web Search using Clickthrough Data》](  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf  )即基于深度网络的语义模型，其核心思想是将query和doc映射到共同维度的语义空间中，通过最大化query和doc语义向量之间的余弦相似度，从而训练得到隐含语义模型，达到检索的目的，并通过word hashing方法来减少输入向量的维度。DSSM有很广泛的应用，比如：搜索引擎检索，广告相关性，问答系统，机器翻译等。

本项目按照论文的网络结构在paddlepaddle上实现DSSM模型，并构造数据集验证网络的正确性。

## 环境

 PaddlePaddle 1.7.0 

 python3.7 

## 单机训练

GPU环境

在gpu_train.sh脚本文件中设置好参数。

```sh
CUDA_VISIBLE_DEVICES=0 python train.py --use_gpu 1 \  #是否使用GPU
                                       --epochs 100 \  #训练轮次
                                       --batch_size 64 \  #batch_size大小
                                       --model_dir "./model_dir" \  #模型保存路径
                                       --TRIGRAM_D 1000 \  #trigram后的向量维度
                                       --L1_N 300 \  #第一层mlp大小
                                       --L2_N 300 \  #第二层mlp大小
                                       --L3_N 128 \  #第三层mlp大小
                                       --Neg 4 \  #负样本采样数量
                                       --base_lr 0.01  #sdg学习率
```

修改脚本的可执行权限并运行

```shell
./gpu_train.sh
```

CPU环境

在cpu_train.sh脚本文件中设置好参数。

```sh
python train.py --use_gpu 0 \  #是否使用GPU
                --epochs 100 \  #训练轮次
                --batch_size 64 \  #batch_size大小
                --model_dir "./model_dir" \  #模型保存路径
                --TRIGRAM_D 1000 \  #trigram后的向量维度
                --L1_N 300 \  #第一层mlp大小
                --L2_N 300 \  #第二层mlp大小
                --L3_N 128 \  #第三层mlp大小
                --Neg 4 \  #负样本采样数量
                --base_lr 0.01  #sdg学习率
```

修改脚本的可执行权限并运行

```
./cpu_train.sh
```

# 预测

在cpu_infer.sh脚本文件中设置好参数。

```sh
python infer.py --use_gpu 0 \  #是否使用GPU
                --batch_size 64 \  #batch_size大小
                --model_dir "./model_dir" \  #模型保存路径
                --TRIGRAM_D 1000 \  #trigram后的向量维度
                --L1_N 300 \  #第一层mlp大小
                --L2_N 300 \  #第二层mlp大小
                --L3_N 128 \  #第三层mlp大小
                --Neg 4 \  #负样本采样数量

```

修改脚本的可执行权限并运行

```sh
./cpu_infer.sh
```

## 模型效果

通过在一个batch中随机构造4个负样本进行训练，可见loss达到收敛状态。

```txt
epoch:1,loss:24.72577
epoch:1,loss:25.79667
epoch:3,loss:22.20090
epoch:3,loss:23.01719
epoch:3,loss:23.71142
epoch:3,loss:22.48990
epoch:3,loss:23.84551
epoch:3,loss:22.21009
epoch:3,loss:22.89378
.......
epoch:12,loss:18.12577
epoch:12,loss:16.66196
epoch:12,loss:17.52151
epoch:12,loss:17.28584
epoch:12,loss:17.33557
epoch:12,loss:16.36964
epoch:12,loss:17.78778
```

