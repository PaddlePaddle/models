# GRU4REC

以下是本例的简要目录结构及说明：

```text
.
├── README.md            # 文档
├── train.py             # 训练脚本
├── infer.py             # 预测脚本
├── utils                # 通用函数
├── convert_format.py    # 转换数据格式
├── small_train.txt      # 小样本训练集
└── small_test.txt       # 小样本测试集

```


## 简介

GRU4REC模型的介绍可以参阅论文[Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)，在本例中，我们实现了GRU4REC的模型。

## RSC15 数据下载及预处理
运行命令 下载RSC15官网数据集
```
curl -Lo yoochoose-data.7z https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z
7z x yoochoose-data.7z
```

GRU4REC的数据过滤，下载脚本[https://github.com/hidasib/GRU4Rec/blob/master/examples/rsc15/preprocess.py](https://github.com/hidasib/GRU4Rec/blob/master/examples/rsc15/preprocess.py)，

注意修改文件路径

line12: PATH_TO_ORIGINAL_DATA = './'

line13:PATH_TO_PROCESSED_DATA = './'

注意使用python3 执行脚本
```
python preprocess.py
```
生成的数据格式如下

```
SessionId    ItemId    Time
1    214536502    1396839069.277
1    214536500    1396839249.868
1    214536506    1396839286.998
1    214577561    1396839420.306
2    214662742    1396850197.614
2    214662742    1396850239.373
2    214825110    1396850317.446
2    214757390    1396850390.71
2    214757407    1396850438.247
```

数据格式需要转换 运行脚本
```
python convert_format.py
```

模型的训练及测试数据如下，一行表示一个用户按照时间顺序的序列

```
214536502 214536500 214536506 214577561
214662742 214662742 214825110 214757390 214757407 214551617
214716935 214774687 214832672
214836765 214706482
214701242 214826623
214826835 214826715
214838855 214838855
214576500 214576500 214576500
214821275 214821275 214821371 214821371 214821371 214717089 214563337 214706462 214717436 214743335 214826837 214819762
214717867 214717867
```

## 训练
'--use_cuda 1' 表示使用gpu, 缺省表示使用cpu '--parallel 1' 表示使用多卡，缺省表示使用单卡

GPU 环境
运行命令 `CUDA_VISIBLE_DEVICES=0 python train.py train_file test_file --use_cuda 1` 开始训练模型。
```
CUDA_VISIBLE_DEVICES=0 python train.py small_train.txt small_test.txt --use_cuda 1
```
CPU 环境
运行命令 `python train.py train_file test_file` 开始训练模型。
```
python train.py small_train.txt small_test.txt
```

当前支持的参数可参见[train.py](./train.py) `train_net` 函数
```python
    batch_size = 50                 # batch大小 推荐500（）
    args = parse_args()  
    vocab, train_reader, test_reader = utils.prepare_data(
        train_file, test_file,batch_size=batch_size * get_cards(args),\
        buffer_size=1000, word_freq_threshold=0)        # buffer_size 局部序列长度排序
    train(
        train_reader=train_reader,  
        vocab=vocab,
        network=network,
        hid_size=100,               # embedding and hidden size
        base_lr=0.01,               # base learning rate
        batch_size=batch_size,
        pass_num=10,                # the number of passed for training
        use_cuda=use_cuda,          # whether to use GPU card
        parallel=parallel,          # whether to be parallel
        model_dir="model_recall20", # directory to save model
        init_low_bound=-0.1,        # uniform parameter initialization lower bound
        init_high_bound=0.1)        # uniform parameter initialization upper bound
```

## 自定义网络结构

可在[train.py](./train.py) `network` 函数中调整网络结构，当前的网络结构如下：
```python
emb = fluid.layers.embedding(
    input=src,
    size=[vocab_size, hid_size],
    param_attr=fluid.ParamAttr(
        initializer=fluid.initializer.Uniform(
            low=init_low_bound, high=init_high_bound),
        learning_rate=emb_lr_x),
    is_sparse=True)

fc0 = fluid.layers.fc(input=emb,
                      size=hid_size * 3,
                      param_attr=fluid.ParamAttr(
                          initializer=fluid.initializer.Uniform(
                              low=init_low_bound, high=init_high_bound),
                          learning_rate=gru_lr_x))
gru_h0 = fluid.layers.dynamic_gru(
    input=fc0,
    size=hid_size,
    param_attr=fluid.ParamAttr(
        initializer=fluid.initializer.Uniform(
            low=init_low_bound, high=init_high_bound),
        learning_rate=gru_lr_x))

fc = fluid.layers.fc(input=gru_h0,
                     size=vocab_size,
                     act='softmax',
                     param_attr=fluid.ParamAttr(
                         initializer=fluid.initializer.Uniform(
                             low=init_low_bound, high=init_high_bound),
                         learning_rate=fc_lr_x))

cost = fluid.layers.cross_entropy(input=fc, label=dst)
acc = fluid.layers.accuracy(input=fc, label=dst, k=20)
```

## 训练结果示例

我们在Tesla K40m单GPU卡上训练的日志如下所示
```text
epoch_1 start
step:100 ppl:441.468
step:200 ppl:311.043
step:300 ppl:218.952
step:400 ppl:186.172
step:500 ppl:188.600
step:600 ppl:131.213
step:700 ppl:165.770
step:800 ppl:164.414
step:900 ppl:156.470
step:1000 ppl:174.201
step:1100 ppl:118.619
step:1200 ppl:122.635
step:1300 ppl:118.220
step:1400 ppl:90.372
step:1500 ppl:135.018
step:1600 ppl:114.327
step:1700 ppl:141.806
step:1800 ppl:93.416
step:1900 ppl:92.897
step:2000 ppl:121.703
step:2100 ppl:96.288
step:2200 ppl:88.355
step:2300 ppl:101.737
step:2400 ppl:95.934
step:2500 ppl:86.158
step:2600 ppl:80.925
step:2700 ppl:202.219
step:2800 ppl:106.828
step:2900 ppl:91.458
step:3000 ppl:105.988
step:3100 ppl:87.067
step:3200 ppl:92.651
step:3300 ppl:101.145
step:3400 ppl:91.247
step:3500 ppl:107.656
step:3600 ppl:89.410
...
...
step:15700 ppl:76.819
step:15800 ppl:62.257
step:15900 ppl:81.735
epoch:1 num_steps:15907 time_cost(s):4154.096032
model saved in model_recall20/epoch_1
...
```

## 预测
运行命令 `CUDA_VISIBLE_DEVICES=0 python infer.py model_dir start_epoch last_epoch(inclusive) train_file test_file` 开始预测.其中，start_epoch指定开始预测的轮次，last_epoch指定结束的轮次，例如
```python
CUDA_VISIBLE_DEVICES=0 python infer.py model 1 10 small_train.txt small_test.txt
```

## 预测结果示例
```text
model:model_r@20/epoch_1 recall@20:0.613 time_cost(s):12.23
model:model_r@20/epoch_2 recall@20:0.647 time_cost(s):12.33
model:model_r@20/epoch_3 recall@20:0.662 time_cost(s):12.38
model:model_r@20/epoch_4 recall@20:0.669 time_cost(s):12.21
model:model_r@20/epoch_5 recall@20:0.673 time_cost(s):12.17
model:model_r@20/epoch_6 recall@20:0.675 time_cost(s):12.26
model:model_r@20/epoch_7 recall@20:0.677 time_cost(s):12.25
model:model_r@20/epoch_8 recall@20:0.679 time_cost(s):12.37
model:model_r@20/epoch_9 recall@20:0.680 time_cost(s):12.22
model:model_r@20/epoch_10 recall@20:0.681 time_cost(s):12.2
```
