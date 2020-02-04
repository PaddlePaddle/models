DyGraph模式下ocr recognition实现
========

简介
--------
ocr任务是识别图片单行的字母信息，在动态图下使用了带attention的seq2seq结构，静态图实现可以参考（[ocr recognition](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/ocr_recognition)）
运行本目录下的程序示例需要使用PaddlePaddle develop最新版本。

动态图文档请见[Dygraph](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/user_guides/howto/dygraph/DyGraph.html)


## 代码结构
```
└── train.py     # 训练脚本。
└── data_reader.py     # 数据读取。
└── utility     # 基础的函数。
```

## 使用的数据

教程中使用`ocr attention`数据集作为训练数据，该数据集通过`paddle.dataset`模块自动下载到本地。

## 训练测试ocr recognition

在GPU单卡上训练ocr recognition:

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

这里`CUDA_VISIBLE_DEVICES=0`表示是执行在0号设备卡上，请根据自身情况修改这个参数。

## 测试ocr recognition


```
CUDA_VISIBLE_DEVICES=0 python eval.py --pretrained_model your_trained_model_path
```

## 预测


```
CUDA_VISIBLE_DEVICES=0 python -u infer.py --pretrained_model your_trained_model_path --image_path your_img_path
```

## 预训练模型

|模型| 准确率|
|- |:-: |
|[ocr_attention_params](https://paddle-ocr-models.bj.bcebos.com/ocr_attention_dygraph.tar) | 82.46%|
