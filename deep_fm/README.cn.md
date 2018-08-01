运行本目录下的程序示例需要使用PaddlePaddle v0.10.0 版本。如果您的PaddlePaddle安装版本低于此要求，请按照[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html)中的说明更新PaddlePaddle安装版本。

---

# 基于深度因子分解机的点击率预估模型

## 介绍
本模型实现了下述论文中提出的DeepFM模型：

```text
@inproceedings{guo2017deepfm,
  title={DeepFM: A Factorization-Machine based Neural Network for CTR Prediction},
  author={Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li and Xiuqiang He},
  booktitle={the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI)},
  pages={1725--1731},
  year={2017}
}
```

DeepFM模型把因子分解机和深度神经网络的低阶和高阶特征的相互作用结合起来，有关因子分解机的详细信息，请参考论文[因子分解机](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)。

## 数据集
本文使用的是Kaggle公司举办的[展示广告竞赛](https://www.kaggle.com/c/criteo-display-ad-challenge/)中所使用的Criteo数据集。

每一行是一次广告展示的特征，第一列是一个标签，表示这次广告展示是否被点击。总共有39个特征，其中13个特征采用整型值，另外26个特征是类别类特征。测试集中是没有标签的。

下载数据集：
```bash
cd data && ./download.sh && cd ..
```

## 模型
DeepFM模型是由因子分解机（FM）和深度神经网络（DNN）组成的。所有的输入特征都会同时输入FM和DNN，最后把FM和DNN的输出结合在一起形成最终的输出。DNN中稀疏特征生成的嵌入层与FM层中的隐含向量（因子）共享参数。

PaddlePaddle中的因子分解机层负责计算二阶组合特征的相互关系。以下的代码示例结合了因子分解机层和全连接层，形成了完整的的因子分解机：

```python
def fm_layer(input, factor_size):
    first_order = paddle.layer.fc(input=input, size=1, act=paddle.activation.Linear())
    second_order = paddle.layer.factorization_machine(input=input, factor_size=factor_size)
    fm = paddle.layer.addto(input=[first_order, second_order],
                            act=paddle.activation.Linear(),
                            bias_attr=False)
    return fm
```

## 数据准备
处理原始数据集，整型特征使用min-max归一化方法规范到[0, 1]，类别类特征使用了one-hot编码。原始数据集分割成两部分：90%用于训练，其他10%用于训练过程中的验证。

```bash
python preprocess.py --datadir ./data/raw --outdir ./data
```

## 训练
训练的命令行选项可以通过`python train.py -h`列出。

训练模型：
```bash
python train.py \
        --train_data_path data/train.txt \
        --test_data_path data/valid.txt \
        2>&1 | tee train.log
```

训练到第9轮的第40000个batch后，测试的AUC为0.807178，误差（cost）为0.445196。

## 预测
预测的命令行选项可以通过`python infer.py -h`列出。

对测试集进行预测：
```bash
python infer.py \
        --model_gz_path models/model-pass-9-batch-10000.tar.gz \
        --data_path data/test.txt \
        --prediction_output_path ./predict.txt
```
