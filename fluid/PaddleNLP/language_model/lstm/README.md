# lstm lm

以下是本例的简要目录结构及说明：

```text
.
├── README.md            # 文档
├── train.py             # 训练脚本
├── reader.py            # 数据读取
└── lm_model.py             # 模型定义文件
```


## 简介

循环神经网络语言模型的介绍可以参阅论文[Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329)，本文主要是说明基于lstm的语言的模型的实现，数据是采用ptb dataset，下载地址为
http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

## 数据下载
用户可以自行下载数据，并解压， 也可以利用目录中的脚本

cd data; sh download_data.sh

## 训练

运行命令
`CUDA_VISIBLE_DEVICES=0 python  train.py --data_path data/simple-examples/data/  --model_type small --use_gpu True`
 开始训练模型。

model_type 为模型配置的大小，目前支持 small，medium, large 三种配置形式

实现采用双层的lstm，具体的参数和网络配置 可以参考 train.py， lm_model.py 文件中的设置


## 训练结果示例

p40中训练日志如下（small config）， test 测试集仅在最后一个epoch完成后进行测试
```text
epoch id 0
ppl  232 865.86505 1.0
ppl  464 632.76526 1.0
ppl  696 510.47153 1.0
ppl  928 437.60617 1.0
ppl  1160 393.38422 1.0
ppl  1392 353.05365 1.0
ppl  1624 325.73267 1.0
ppl  1856 305.488 1.0
ppl  2088 286.3128 1.0
ppl  2320 270.91504 1.0
train ppl 270.86246
valid ppl 181.867964379
...
ppl  2320 40.975872 0.001953125
train ppl 40.974102
valid ppl 117.85741214
test ppl 113.939103843
```
## 与tf结果对比

tf采用的版本是1.6
```text
small config
             train    valid       test
fluid 1.0   40.962    118.111     112.617
tf 1.6      40.492    118.329     113.788

medium config
             train    valid      test  
fluid 1.0   45.620   87.398      83.682
tf 1.6      45.594   87.363      84.015

large config
             train    valid      test
fluid 1.0   37.221   82.358      78.137
tf 1.6      38.342   82.311      78.121
```
