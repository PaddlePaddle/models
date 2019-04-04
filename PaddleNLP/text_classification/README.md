# 文本分类

以下是本例的简要目录结构及说明：

```text
.
├── nets.py              # 模型定义
├── README.md            # 文档
├── train.py             # 训练脚本
├── infer.py             # 预测脚本
└── utils.py             # 定义通用函数，从外部获取
```


## 简介，模型详解

在PaddlePaddle v2版本[文本分类](https://github.com/PaddlePaddle/models/blob/develop/legacy/text_classification/README.md)中对于文本分类任务有较详细的介绍，在本例中不再重复介绍。
在模型上，我们采用了bow, cnn, lstm, gru四种常见的文本分类模型。

## 训练

1. 运行命令 `python train.py bow` 开始训练模型。
    ```python
    python train.py bow    # bow指定网络结构，可替换成cnn, lstm, gru
    ```

2. (可选）想自定义网络结构，需在[nets.py](./nets.py)中自行添加，并设置[train.py](./train.py)中的相应参数。
    ```python
    def train(train_reader,     # 训练数据
        word_dict,              # 数据字典
        network,                # 模型配置
        use_cuda,               # 是否用GPU
        parallel,               # 是否并行
        save_dirname,           # 保存模型路径
        lr=0.2,                 # 学习率大小
        batch_size=128,         # 每个batch的样本数
        pass_num=30):           # 训练的轮数
    ```

## 训练结果示例
```text
    pass_id: 0, avg_acc: 0.848040, avg_cost: 0.354073
    pass_id: 1, avg_acc: 0.914200, avg_cost: 0.217945
    pass_id: 2, avg_acc: 0.929800, avg_cost: 0.184302
    pass_id: 3, avg_acc: 0.938680, avg_cost: 0.164240
    pass_id: 4, avg_acc: 0.945120, avg_cost: 0.149150
    pass_id: 5, avg_acc: 0.951280, avg_cost: 0.137117
    pass_id: 6, avg_acc: 0.955360, avg_cost: 0.126434
    pass_id: 7, avg_acc: 0.961400, avg_cost: 0.117405
    pass_id: 8, avg_acc: 0.963560, avg_cost: 0.110070
    pass_id: 9, avg_acc: 0.965840, avg_cost: 0.103273
    pass_id: 10, avg_acc: 0.969800, avg_cost: 0.096314
    pass_id: 11, avg_acc: 0.971720, avg_cost: 0.090206
    pass_id: 12, avg_acc: 0.974800, avg_cost: 0.084970
    pass_id: 13, avg_acc: 0.977400, avg_cost: 0.078981
    pass_id: 14, avg_acc: 0.980000, avg_cost: 0.073685
    pass_id: 15, avg_acc: 0.981080, avg_cost: 0.069898
    pass_id: 16, avg_acc: 0.982080, avg_cost: 0.064923
    pass_id: 17, avg_acc: 0.984680, avg_cost: 0.060861
    pass_id: 18, avg_acc: 0.985840, avg_cost: 0.057095
    pass_id: 19, avg_acc: 0.988080, avg_cost: 0.052424
    pass_id: 20, avg_acc: 0.989160, avg_cost: 0.049059
    pass_id: 21, avg_acc: 0.990120, avg_cost: 0.045882
    pass_id: 22, avg_acc: 0.992080, avg_cost: 0.042140
    pass_id: 23, avg_acc: 0.992280, avg_cost: 0.039722
    pass_id: 24, avg_acc: 0.992840, avg_cost: 0.036607
    pass_id: 25, avg_acc: 0.994440, avg_cost: 0.034040
    pass_id: 26, avg_acc: 0.995000, avg_cost: 0.031501
    pass_id: 27, avg_acc: 0.995440, avg_cost: 0.028988
    pass_id: 28, avg_acc: 0.996240, avg_cost: 0.026639
    pass_id: 29, avg_acc: 0.996960, avg_cost: 0.024186
```

## 预测
1. 运行命令 `python infer.py bow_model`, 开始预测。
    ```python
    python infer.py bow_model     # bow_model指定需要导入的模型

## 预测结果示例
```text
    model_path: bow_model/epoch0, avg_acc: 0.882800
    model_path: bow_model/epoch1, avg_acc: 0.882360
    model_path: bow_model/epoch2, avg_acc: 0.881400
    model_path: bow_model/epoch3, avg_acc: 0.877800
    model_path: bow_model/epoch4, avg_acc: 0.872920
    model_path: bow_model/epoch5, avg_acc: 0.872640
    model_path: bow_model/epoch6, avg_acc: 0.869960
    model_path: bow_model/epoch7, avg_acc: 0.865160
    model_path: bow_model/epoch8, avg_acc: 0.863680
    model_path: bow_model/epoch9, avg_acc: 0.861200
    model_path: bow_model/epoch10, avg_acc: 0.853520
    model_path: bow_model/epoch11, avg_acc: 0.850400
    model_path: bow_model/epoch12, avg_acc: 0.855960
    model_path: bow_model/epoch13, avg_acc: 0.853480
    model_path: bow_model/epoch14, avg_acc: 0.855960
    model_path: bow_model/epoch15, avg_acc: 0.854120
    model_path: bow_model/epoch16, avg_acc: 0.854160
    model_path: bow_model/epoch17, avg_acc: 0.852240
    model_path: bow_model/epoch18, avg_acc: 0.852320
    model_path: bow_model/epoch19, avg_acc: 0.850280
    model_path: bow_model/epoch20, avg_acc: 0.849760
    model_path: bow_model/epoch21, avg_acc: 0.850160
    model_path: bow_model/epoch22, avg_acc: 0.846800
    model_path: bow_model/epoch23, avg_acc: 0.845440
    model_path: bow_model/epoch24, avg_acc: 0.845640
    model_path: bow_model/epoch25, avg_acc: 0.846200
    model_path: bow_model/epoch26, avg_acc: 0.845880
    model_path: bow_model/epoch27, avg_acc: 0.844880
    model_path: bow_model/epoch28, avg_acc: 0.844680
    model_path: bow_model/epoch29, avg_acc: 0.844960
```
注：过拟合导致acc持续下降，请忽略
