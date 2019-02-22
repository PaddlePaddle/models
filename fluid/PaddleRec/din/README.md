# DIN

以下是本例的简要目录结构及说明：

```text
.
├── README.md            # 文档
├── train.py             # 训练脚本
├── infer.py             # 预测脚本
├── network.py           # 网络结构
├── cluster_train.py     # 多机训练
├── cluster_train.sh     # 多机训练脚本
├── reader.py            # 和读取数据相关的函数
├── data/
    ├── build_dataset.py    # 文本数据转化为paddle数据
    ├── convert_pd.py       # 将原始数据转化为pandas的dataframe
    ├── data_process.sh     # 数据预处理脚本
    ├── remap_id.py         # remap类别id

```

## 简介

DIN模型的介绍可以参阅论文[Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)。

DIN通过一个兴趣激活模块(Activation Unit)，用预估目标Candidate ADs的信息去激活用户的历史点击商品，以此提取用户与当前预估目标相关的兴趣。

权重高的历史行为表明这部分兴趣和当前广告相关，权重低的则是和广告无关的”兴趣噪声“。我们通过将激活的商品和激活权重相乘，然后累加起来作为当前预估目标ADs相关的兴趣状态表达。

最后我们将这相关的用户兴趣表达、用户静态特征和上下文相关特征，以及ad相关的特征拼接起来，输入到后续的多层DNN网络，最后预测得到用户对当前目标ADs的点击概率。

我们复现了论文效果，best auc可以达到0.87


## 数据下载及预处理

* Step 1: 运行如下命令 下载Amazon Product数据集并进行预处理
```
cd data && sh data_process.sh && cd ..
```

* Step 2: 产生训练集、测试集和config文件
```
python build_dataset.py
```
运行之后在data文件夹下会产生config.txt、paddle_test.txt、paddle_train.txt三个文件


## 训练

具体的参数配置说明可通过运行下列代码查看
```
python train.py -h
```

gpu 单机单卡训练
``` bash
CUDA_VISIBLE_DEVICES=1 python -u train.py --config_path 'data/config.txt' --train_dir 'data/paddle_train.txt' --batch_size 32 --epoch_num 100 --use_cuda 1 > log.txt 2>&1 &
```

cpu 单机训练
``` bash
python -u train.py --config_path 'data/config.txt' --train_dir 'data/paddle_train.txt' --batch_size 32 --epoch_num 100 --use_cuda 0 > log.txt 2>&1 &
```

值得注意的是上述单卡训练可以通过加--parallel 1参数使用Parallel Executor来进行加速

gpu 单机多卡训练
``` bash
CUDA_VISIBLE_DEVICES=0,1 python -u train.py --config_path 'data/config.txt' --train_dir 'data/paddle_train.txt' --batch_size 32 --epoch_num 100 --use_cuda 0 --parallel 0 > log.txt 2>&1 &
```

cpu 单机多卡训练
``` bash
CPU_NUM=10 python -u train.py --config_path 'data/config.txt' --train_dir 'data/paddle_train.txt' --batch_size 32 --epoch_num 100 --use_cuda 0 --parallel 1 > log.txt 2>&1 &
```


## 训练结果示例

我们在Tesla K40m单GPU卡上训练的日志如下所示(以实际输出为准)
```text
2019-02-22 09:31:51,578 - INFO - reading data begins
2019-02-22 09:32:22,407 - INFO - reading data completes
W0222 09:32:24.151955  7221 device_context.cc:263] Please NOTE: device: 0, CUDA Capability: 35, Driver API Version: 9.0, Runtime API Version: 8.0
W0222 09:32:24.152046  7221 device_context.cc:271] device: 0, cuDNN Version: 7.0.
2019-02-22 09:32:27,797 - INFO - train begins
epoch: 1    global_step: 1000    train_loss: 0.6950        time: 14.64
epoch: 1    global_step: 2000    train_loss: 0.6854        time: 15.41
epoch: 1    global_step: 3000    train_loss: 0.6799        time: 14.84
...
model saved in din_amazon/global_step_50000
...
```

## 预测
运行命令 开始预测.

```
CUDA_VISIBLE_DEVICES=3 python infer.py --model_path 'din_amazon/global_step_400000' --test_path 'data/paddle_test.txt' --use_cuda 1
```

## 预测结果示例
```text
2019-02-22 11:22:58,804 - INFO - TEST --> loss: [0.47005194] auc:0.863794952818
```


## 多机训练
厂内用户可以参考[wiki](http://wiki.baidu.com/pages/viewpage.action?pageId=628300529)利用paddlecloud 配置多机环境

可参考cluster_train.py 配置其他多机环境

运行命令本地模拟多机场景
```
sh cluster_train.sh
```

注意本地模拟需要关闭代理
