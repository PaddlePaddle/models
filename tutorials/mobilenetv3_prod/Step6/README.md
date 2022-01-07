# MobileNetV3

## 目录


- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
    - [3.1 准备环境]()
    - [3.2 准备数据]()
    - [3.3 准备模型]()
- [4. 开始使用]()
    - [4.1 模型训练]()
    - [4.2 模型评估]()
    - [4.3 模型预测]()
- [5. 模型推理部署]()
- [6. TIPC自动化测试脚本]()
- [7. 参考链接与文献]()


## 1. 简介

* coming soon!

**论文:** [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

**参考repo:** [https://github.com/pytorch/vision](https://github.com/pytorch/vision)


在此感谢[vision](https://github.com/pytorch/vision)，提高了MobileNetV3论文复现的效率。

注意：在这里为了简化流程，仅关于`ImageNet标准训练过程`做训练对齐，具体地：
* 训练总共120epoch，总的batch size是256*8=2048，学习率为0.8，下降策略为Piecewise Decay(30epoch下降10倍)
* 训练预处理：RandomResizedCrop(size=224) + RandomFlip(p=0.5) + Normalize
* 评估预处理：Resize(256) + CenterCrop(224) + Normalize

这里`mobilenet_v3_small`的参考指标也是重新训练得到的。

## 2. 数据集和复现精度

数据集为ImageNet，训练集包含1281167张图像，验证集包含50000张图像。

您可以从[ImageNet 官网](https://image-net.org/)申请下载数据。


| 模型      | top1/5 acc (参考精度) | top1/5 acc (复现精度) | 下载链接 |
|:---------:|:------:|:----------:|:----------:|
| Mo | -/-   | 0.601/0.826   | [预训练模型](https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/mobilenet_v3_small_pretrained.pdparams) \|  [Inference模型(coming soon!)]() \| [日志](https://paddle-model-ecology.bj.bcebos.com/model/mobilenetv3_reprod/train_mobilenet_v3_small.log) |


## 3. 准备环境与数据

### 3.1 准备环境

* 下载代码

```bash
https://github.com/PaddlePaddle/models.git
cd model/tutorials/mobilenetv3_prod/Step6
```

* 安装paddlepaddle

```bash
# 需要安装2.2及以上版本的Paddle，如果
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0
# 安装CPU版本的Paddle
pip install paddlepaddle==2.2.0
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 准备数据

如果您已经下载好ImageNet1k数据集，那么该步骤可以跳过，如果您没有，则可以从[ImageNet官网](https://image-net.org/download.php)申请下载。

如果只是希望快速体验模型训练功能，则可以直接解压`test_images/lite_data.tar`，其中包含16张训练图像以及16张验证图像。

```bash
tar -xf test_images/lite_data.tar
```

### 3.3 准备模型

如果您希望直接体验评估或者预测推理过程，可以直接根据第2章的内容下载提供的预训练模型，直接体验模型评估、预测、推理部署等内容。


## 4. 开始使用

### 4.1 模型训练

* 单机单卡训练

```bash
export CUDA_VISIBLE_DEVICES=0
python3.7 train.py --data-path=./ILSVRC2012 --lr=0.1 --batch-size=256
```

部分训练日志如下所示。

```
[Epoch 1, iter: 4780] top1: 0.10312, top5: 0.27344, lr: 0.01000, loss: 5.34719, avg_reader_cost: 0.03644 sec, avg_batch_cost: 0.05536 sec, avg_samples: 64.0, avg_ips: 1156.08863 images/sec.
[Epoch 1, iter: 4790] top1: 0.08750, top5: 0.24531, lr: 0.01000, loss: 5.28853, avg_reader_cost: 0.05164 sec, avg_batch_cost: 0.06852 sec, avg_samples: 64.0, avg_ips: 934.08427 images/sec.
```

* 单机多卡训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" train.py --data-path="./ILSVRC2012" --lr=0.4 --batch-size=256
```

更多配置参数可以参考[train.py](./train.py)的`get_args_parser`函数。

### 4.2 模型评估

该项目中，训练与评估脚本相同，指定`--test-only`参数即可完成预测过程。

```bash
python train.py --test-only --data-path=/paddle/data/ILSVRC2012 --pretrained=./mobilenet_v3_small_paddle.pdparams
```

期望输出如下。

```
Test:  [   0/1563]  eta: 1:14:20  loss: 1.0456 (1.0456)  acc1: 0.7812 (0.7812)  acc5: 0.9062 (0.9062)  time: 2.8539  data: 2.8262
...
Test:  [1500/1563]  eta: 0:00:05  loss: 1.2878 (1.9196)  acc1: 0.7344 (0.5639)  acc5: 0.8750 (0.7893)  time: 0.0623  data: 0.0534
Test: Total time: 0:02:05
 * Acc@1 0.564 Acc@5 0.790
```

### 4.3 模型预测

* 使用GPU预测

```
python tools/predict.py --pretrained=./mobilenet_v3_small_paddle_pretrained.pdparams --img-path=images/demo.jpg
```

对于下面的图像进行预测

<div align="center">
    <img src="./images/demo.jpg" width=300">
</div>

最终输出结果为`class_id: 8, prob: 0.9091238975524902`，表示预测的类别ID是`8`，置信度为`0.909`。

* 使用CPU预测

```
python tools/predict.py --pretrained=./mobilenet_v3_small_paddle_pretrained.pdparams --img-path=images/demo.jpg --device=cpu
```


## 5. 模型推理部署

coming soon!


## 6. TIPC自动化测试脚本

coming soon!

## 7. 参考链接与文献

1. Howard A, Sandler M, Chu G, et al. Searching for mobilenetv3[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 1314-1324.
2. vision: https://github.com/pytorch/vision
