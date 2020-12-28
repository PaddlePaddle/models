# Language Model

## Transformer-XL

以下是本例的简要目录结构及说明：

```text
.
├── eval.py                 # 预测脚本
├── reader.py               # 数据读取接口
├── README.md               # 文档
├── train.py                # 训练脚本
└── configs                 # 配置文件
```

## 模型简介

本项目是语言模型 Transformer-XL 的 PaddlePaddle 实现， 包含模型训练，预测等内容。


## 快速开始

### 安装说明

1. paddle安装

    本项目依赖于 PaddlePaddle 2.0rc及以上版本或适当的develop版本，请参考 [安装指南](https://www.paddlepaddle.org.cn/install/quick) 进行安装

2. 下载代码

    克隆代码库到本地

3. 环境依赖

    该模型使用PaddlePaddle，关于环境依赖部分，请先参考PaddlePaddle[安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)关于环境依赖部分的内容。
    此外，需要另外涉及：
      * attrdict
      * pyyaml



### 数据准备

公开数据集：enwik8、text8、wt103 多用于语言模型的 benchmark 测试。输出获取与处理方式如下：

```shell
bash gen_data.sh
```

会在当前路径下的 ./gen_data/ 路径下生成我们需要的数据。


### 单机训练

### 单机单卡

以提供的 enwik8 数据为例，可以执行以下命令进行模型训练：

```sh
# setting visible devices for training
export CUDA_VISIBLE_DEVICES=0
python train.py --config ./configs/enwik8.yaml
```

可以在 enwik8.yaml 文件中设置相应的参数，比如 `batch_size`、`epoch` 等。

### 单机多卡

同样，可以执行如下命令实现八卡训练：

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train.py --config ./configs/enwik8.yaml
```


### 模型推断

以 enwik8 数据为例，模型训练完成后可以执行以下命令可以进行预测：

```sh
# setting visible devices for prediction
export CUDA_VISIBLE_DEVICES=0
python eval.py --config ./configs/enwik8.yaml
```

完成推断之后，会将显示在验证集和测试集上的结果。

## 参考文献
