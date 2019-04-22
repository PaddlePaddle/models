# 语言模型

以下是本例的简要目录结构及说明：

```text
.
├── README.md            # 文档
├── train.py             # 训练脚本
├── infer.py             # 预测脚本
└── utils.py             # 通用函数
```


## 简介

循环神经网络语言模型的介绍可以参阅论文[Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329)，在本例中，我们实现了GRU-RNN语言模型。

## 训练

运行命令 `python train.py` 开始训练模型。
```python
python train.py
```

当前支持的参数可参见[train.py](./train.py) `train_net` 函数
```python
vocab, train_reader, test_reader = utils.prepare_data(
        batch_size=20, # batch size
        buffer_size=1000, # buffer size, default value is OK
        word_freq_threshold=0) # vocabulary related parameter, and words with frequency below this value will be filtered

train(train_reader=train_reader,
        vocab=vocab,
        network=network,
        hid_size=200, # embedding and hidden size
        base_lr=1.0, # base learning rate
        batch_size=20, # batch size, the same as that in prepare_data
        pass_num=12, # the number of passes for training
        use_cuda=True, # whether to use GPU card
        parallel=False, # whether to be parallel
        model_dir="model", # directory to save model
        init_low_bound=-0.1, # uniform parameter initialization lower bound
        init_high_bound=0.1) # uniform parameter initialization upper bound
```

## 自定义网络结构

可在[train.py](./train.py) `network` 函数中调整网络结构，当前的网络结构如下：
```python
emb = fluid.layers.embedding(input=src, size=[vocab_size, hid_size],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(low=init_low_bound, high=init_high_bound),
            learning_rate=emb_lr_x),
        is_sparse=True)

fc0 = fluid.layers.fc(input=emb, size=hid_size * 3,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(low=init_low_bound, high=init_high_bound),
            learning_rate=gru_lr_x))
gru_h0 = fluid.layers.dynamic_gru(input=fc0, size=hid_size,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(low=init_low_bound, high=init_high_bound),
            learning_rate=gru_lr_x))

fc = fluid.layers.fc(input=gru_h0, size=vocab_size, act='softmax',
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(low=init_low_bound, high=init_high_bound),
            learning_rate=fc_lr_x))

cost = fluid.layers.cross_entropy(input=fc, label=dst)
```

## 训练结果示例

我们在Tesla K40m单GPU卡上训练的日志如下所示
```text
epoch_1 start
step:100 ppl:771.053
step:200 ppl:449.597
step:300 ppl:642.654
step:400 ppl:458.128
step:500 ppl:510.912
step:600 ppl:451.545
step:700 ppl:364.404
step:800 ppl:324.272
step:900 ppl:360.797
step:1000 ppl:275.761
step:1100 ppl:294.599
step:1200 ppl:335.877
step:1300 ppl:185.262
step:1400 ppl:241.744
step:1500 ppl:211.507
step:1600 ppl:233.431
step:1700 ppl:298.767
step:1800 ppl:203.403
step:1900 ppl:158.828
step:2000 ppl:171.148
step:2100 ppl:280.884
epoch:1 num_steps:2104 time_cost(s):47.478780
model saved in model/epoch_1
epoch_2 start
step:100 ppl:238.099
step:200 ppl:136.527
step:300 ppl:204.184
step:400 ppl:252.886
step:500 ppl:177.377
step:600 ppl:197.688
step:700 ppl:131.650
step:800 ppl:223.906
step:900 ppl:144.785
step:1000 ppl:176.286
step:1100 ppl:148.158
step:1200 ppl:203.581
step:1300 ppl:168.208
step:1400 ppl:159.412
step:1500 ppl:114.032
step:1600 ppl:157.985
step:1700 ppl:147.743
step:1800 ppl:88.676
step:1900 ppl:141.962
step:2000 ppl:106.087
step:2100 ppl:122.709
epoch:2 num_steps:2104 time_cost(s):47.583789
model saved in model/epoch_2
...
```

## 预测
运行命令 `python infer.py model_dir start_epoch last_epoch(inclusive)` 开始预测，其中，start_epoch指定开始预测的轮次，last_epoch指定结束的轮次，例如
```python
python infer.py model 1 12 # prediction from epoch 1 to epoch 12
```

## 预测结果示例
```text
model:model/epoch_1 ppl:254.540 time_cost(s):3.29
model:model/epoch_2 ppl:177.671 time_cost(s):3.27
model:model/epoch_3 ppl:156.251 time_cost(s):3.27
model:model/epoch_4 ppl:139.036 time_cost(s):3.27
model:model/epoch_5 ppl:132.661 time_cost(s):3.27
model:model/epoch_6 ppl:130.092 time_cost(s):3.28
model:model/epoch_7 ppl:128.751 time_cost(s):3.27
model:model/epoch_8 ppl:125.411 time_cost(s):3.27
model:model/epoch_9 ppl:124.604 time_cost(s):3.28
model:model/epoch_10 ppl:124.754 time_cost(s):3.29
model:model/epoch_11 ppl:125.421 time_cost(s):3.27
model:model/epoch_12 ppl:125.676 time_cost(s):3.27
```
