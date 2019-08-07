# YOLOv3 目标检测

---
## 内容

- [简介](#简介)
- [快速开始](#快速开始)
- [进阶使用](#进阶使用)
- [FAQ](#faq)
- [参考文献](#参考文献)
- [版本更新](#版本更新)
- [如何贡献代码](#如何贡献代码)
- [作者](#作者)

## 简介

[YOLOv3](https://arxiv.org/abs/1804.02767) 是由 [Joseph Redmon](https://arxiv.org/search/cs?searchtype=author&query=Redmon%2C+J) 和 [Ali Farhadi](https://arxiv.org/search/cs?searchtype=author&query=Farhadi%2C+A) 提出的单阶段检测器, 该检测器与达到同样精度的传统目标检测方法相比，推断速度能达到接近两倍.

在我们的实现版本中使用了 [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/abs/1902.04103v3) 中提出的图像增强和label smooth等优化方法，精度优于darknet框架的实现版本，在COCO-2017数据集上，我们达到`mAP(0.50:0.95)= 38.9`的精度，比darknet实现版本的精度(33.0)要高5.9.

同时，在推断速度方面，基于Paddle预测库的加速方法，推断速度比darknet高30%.

## 快速开始

### 安装

**安装[COCO-API](https://github.com/cocodataset/cocoapi)：**

训练前需要首先下载[COCO-API](https://github.com/cocodataset/cocoapi)：

    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    # if cython is not installed
    pip install Cython
    # Install into global site-packages
    make install
    # Alternatively, if you do not have permissions or prefer
    # not to install the COCO API into global site-packages
    python2 setup.py install --user

**安装[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)：**

在当前目录下运行样例代码需要PadddlePaddle Fluid的v.1.4或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html)中的说明来更新PaddlePaddle。

### 数据准备

**COCO数据集：**

在[MS-COCO数据集](http://cocodataset.org/#download)上进行训练，通过如下方式下载数据集。

    cd dataset/coco
    ./download.sh

数据目录结构如下：

```
dataset/coco/
├── annotations
│   ├── instances_train2014.json
│   ├── instances_train2017.json
│   ├── instances_val2014.json
│   ├── instances_val2017.json
|   ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000580008.jpg
|   ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
|   ...

```

**自定义数据集：**

用户可使用自定义的数据集，我们推荐自定义数据集使用COCO数据集格式的标注，并可通过设置`--data_dir`或修改[reader.py](./reader.py#L39)指定数据集路径。使用COCO数据集格式标注时，目录结构可参考上述COCO数据集目录结构。

### 模型训练

**下载预训练模型：** 本示例提供DarkNet-53预训练[模型](https://paddlemodels.bj.bcebos.com/yolo/darknet53.tar.gz)，该模型转换自作者提供的预训练权重[pjreddie/darknet](https://pjreddie.com/media/files/darknet53.conv.74)，采用如下命令下载预训练模型：

    sh ./weights/download.sh

通过设置`--pretrain` 加载预训练模型。同时在fine-tune时也采用该设置加载已训练模型。
请在训练前确认预训练模型下载与加载正确，否则训练过程中损失可能会出现NAN。

**开始训练：** 数据准备完毕后，可以通过如下的方式启动训练：

    python train.py \
       --model_save_dir=output/ \
       --pretrain=${path_to_pretrain_model} \
       --data_dir=${path_to_data} \
       --class_num=${category_num}

- 通过设置`export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`指定8卡GPU训练。
- 若在Windows环境下训练模型，建议设置`--use_multiprocess_reader=False`。
- 通过`--worker_num=`设置多进程数据读取器进程数，默认进程数为8，若训练机器CPU核数较少，建议设小该值。
- 可选参数见：

    python train.py --help

**注意：** YOLOv3模型总batch size为64，这里使用8 GPUs每GPU上batch size为8来训练

**模型设置：**

*  模型使用了基于COCO数据集生成的9个先验框：10x13，16x30，33x23，30x61，62x45，59x119，116x90，156x198，373x326
*  YOLOv3模型中，若预测框不是该点最佳匹配框但是和任一ground truth框的重叠大于`ignore_thresh=0.7`，则忽略该预测框的目标性损失

**训练策略：**

*  采用momentum优化算法训练YOLOv3，momentum=0.9。
*  学习率采用warmup算法，前4000轮学习率从0.0线性增加至0.001。在400000，450000轮时使用0.1,0.01乘子进行学习率衰减，最大训练500000轮。
*  通过设置`--syncbn=True`可以开启Synchronized batch normalization，该模式下精度会提高

**注意：** Synchronized batch normalization只能用于多GPU训练，不能用于CPU训练和单GPU训练。

下图为模型训练结果：
<p align="center">
<img src="image/train_loss.png" height="400" width="550" hspace="10"/><br />
Train Loss
</p>

### 模型评估

模型评估是指对训练完毕的模型评估各类性能指标。本示例采用[COCO官方评估](http://cocodataset.org/#detections-eval), 用户可通过如下方式下载Paddle发布的YOLOv3[模型](https://paddlemodels.bj.bcebos.com/yolo/yolov3.tar.gz)

    sh ./weights/download.sh

`eval.py`是评估模块的主要执行程序，调用示例如下：

    python eval.py \
        --dataset=coco2017 \
        --weights=${path_to_weights} \
        --class_num=${category_num}

- 通过设置`export CUDA_VISIBLE_DEVICES=0`指定单卡GPU评估。

若训练时指定`--syncbn=False`, 模型评估精度如下:

|   input size  | mAP(IoU=0.50:0.95) | mAP(IoU=0.50) | mAP(IoU=0.75) |
| :------: | :------: | :------: | :------: |
| 608x608 | 37.7 | 59.8 | 40.8 |
| 416x416 | 36.5 | 58.2 | 39.1 |
| 320x320 | 34.1 | 55.4 | 36.3 |

若训练时指定`--syncbn=True`, 模型评估精度如下:

|   input size  | mAP(IoU=0.50:0.95) | mAP(IoU=0.50) | mAP(IoU=0.75) |
| :------: | :------: | :------: | :------: |
| 608x608 | 38.9 | 61.1 | 42.0 |
| 416x416 | 37.5 | 59.6 | 40.2 |
| 320x320 | 34.8 | 56.4 | 36.9 |

- **注意：** 评估结果基于`pycocotools`评估器，没有滤除`score < 0.05`的预测框，其他框架有此滤除操作会导致精度下降。

### 模型推断及可视化

模型推断可以获取图像中的物体及其对应的类别，`infer.py`是主要执行程序，调用示例如下：

    python infer.py \
       --dataset=coco2017 \
        --weights=${path_to_weights}  \
        --class_num=${category_num} \
        --image_path=data/COCO17/val2017/  \
        --image_name=000000000139.jpg \
        --draw_thresh=0.5

- 通过设置`export CUDA_VISIBLE_DEVICES=0`指定单卡GPU预测。
- 推断结果显示如下，并会在`./output`目录下保存带预测框的图像

```
Image person.jpg detect:
   person          at [190, 101, 273, 372]      score: 0.98832
   dog             at [63, 263, 200, 346]       score: 0.97049
   horse           at [404, 137, 598, 366]      score: 0.97305
Detect result save at ./output/person.png
```

下图为模型可视化预测结果：
<p align="center">
<img src="image/000000000139.png" height=300 width=400 hspace='10'/>
<img src="image/000000127517.png" height=300 width=400 hspace='10'/>
<img src="image/000000203864.png" height=300 width=400 hspace='10'/>
<img src="image/000000515077.png" height=300 width=400 hspace='10'/> <br />
YOLOv3 预测可视化
</p>

### Benchmark

模型训练benchmark:

| 数据集 | GPU | CUDA | cuDNN | batch size | 训练速度(1 GPU) | 训练速度(8 GPU) | 显存占用(1 GPU) | 显存占用(8 GPU) |
| :-----: | :-: | :--: | :---: | :--------: | :-----------------: | :-----------------: | :------------: | :------------: |
| COCO | Tesla P40 | 8.0 | 7.1 | 8 (per GPU) | 30.2 images/s | 59.3 images/s | 10642 MB/GPU | 10782 MB/GPU |

模型单卡推断速度：

| GPU | CUDA | cuDNN | batch size | infer speed(608x608) | infer speed(416x416) | infer speed(320x320) |
| :-: | :--: | :---: | :--------: | :-----: | :-----: | :-----: |
| Tesla P40 | 8.0 | 7.1 | 1 | 48 ms/frame | 29 ms/frame |24 ms/frame |

### 服务部署

进行YOLOv3的服务部署，用户可以在[eval.py](./eval.py#L54)或[infer.py](./infer.py#L47)中保存可部署的推断模型，该模型可以用Paddle预测库加载和部署，参考[Paddle预测库](http://paddlepaddle.org/documentation/docs/zh/1.4/advanced_usage/deploy/index_cn.html)

## 进阶使用

### 背景介绍

传统目标检测方法通过两阶段检测，第一阶段生成预选框，第二阶段对预选框进行分类得到类别，而YOLO将目标检测看做是对框位置和类别概率的一个单阶段回归问题，使得YOLO能达到近两倍的检测速度。而YOLOv3在YOLO的基础上引入的多尺度预测，使得YOLOv3网络对于小物体的检测精度大幅提高。

### 模型概览

[YOLOv3](https://arxiv.org/abs/1804.02767) 是一阶段End2End的目标检测器。其目标检测原理如下图所示:
<p align="center">
<img src="image/YOLOv3.jpg" height=400 width=600 hspace='10'/> <br />
YOLOv3检测原理
</p>

### 模型结构

YOLOv3将输入图像分成S\*S个格子，每个格子预测B个bounding box，每个bounding box预测内容包括: Location(x, y, w, h)、Confidence Score和C个类别的概率，因此YOLOv3输出层的channel数为S\*S\*B\*(5 + C)。YOLOv3的loss函数也有三部分组成：Location误差，Confidence误差和分类误差。

YOLOv3的网络结构如下图所示:
<p align="center">
<img src="image/YOLOv3_structure.jpg" height=400 width=400 hspace='10'/> <br />
YOLOv3网络结构
</p>

YOLOv3 的网络结构由基础特征提取网络、multi-scale特征融合层和输出层组成。

1. 特征提取网络。YOLOv3使用 [DarkNet53](https://arxiv.org/abs/1612.08242)作为特征提取网络：DarkNet53 基本采用了全卷积网络，用步长为2的卷积操作替代了池化层，同时添加了 Residual 单元，避免在网络层数过深时发生梯度弥散。

2. 特征融合层。为了解决之前YOLO版本对小目标不敏感的问题，YOLOv3采用了3个不同尺度的特征图来进行目标检测，分别为13\*13,26\*26,52\*52,用来检测大、中、小三种目标。特征融合层选取 DarkNet 产出的三种尺度特征图作为输入，借鉴了FPN(feature pyramid networks)的思想，通过一系列的卷积层和上采样对各尺度的特征图进行融合。

3. 输出层。同样使用了全卷积结构，其中最后一个卷积层的卷积核个数是255：3\*(80+4+1)=255，3表示一个grid cell包含3个bounding box，4表示框的4个坐标信息，1表示Confidence Score，80表示COCO数据集中80个类别的概率。

### 模型fine-tune

对YOLOv3进行fine-tune，用户可用`--pretrain`指定下载好的Paddle发布的YOLOv3[模型](https://paddlemodels.bj.bcebos.com/yolo/yolov3.tar.gz)，并把`--class_num`设置为用户数据集的类别数。

在fine-tune时，若用户自定义数据集的类别数不等于COCO数据集的80类，则加载权重时不应加载`yolo_output`层的权重，可通过在[train.py](./train.py#L76)使用如下方式加载非`yolo_output`层的权重：

```python
if cfg.pretrain:
    if not os.path.exists(cfg.pretrain):
        print("Pretrain weights not found: {}".format(cfg.pretrain))

    def if_exist(var):
        return os.path.exists(os.path.join(cfg.pretrain, var.name)) \
               and var.name.find('yolo_output') < 0

    fluid.io.load_vars(exe, cfg.pretrain, predicate=if_exist)

```

若用户自定义数据集的类别是COCO数据集类别的子集，`yolo_output`层的权重可以进行裁剪后导入。例如用户数据集有6类分别对应COCO数据集80类中的第`[3, 19, 25, 41, 58, 73]`类，可通过如下方式裁剪`yolo_output`层权重：

```python
if cfg.pretrain:
    if not os.path.exists(cfg.pretrain):
        print("Pretrain weights not found: {}".format(cfg.pretrain))

    def if_exist(var):
        return os.path.exists(os.path.join(cfg.pretrain, var.name))

    fluid.io.load_vars(exe, cfg.pretrain, predicate=if_exist)

    cat_idxs = [3, 19, 25, 41, 58, 73]
    # the first 5 channels is x, y, w, h, objectness, 
    # the following 80 channel is for 80 categories
    channel_idxs = np.array(range(5) + [idx + 5 for idx in cat_idxs])
    # we have 3 yolo_output layers
    for i in range(3): 
        # crop conv weights
        weights_tensor = fluid.global_scope().find_var(
                          "yolo_output.{}.conv.weights".format(i)).get_tensor()
        weights = np.array(weights_tensor)
        # each yolo_output layer has 3 anchors, 85 channels of each anchor
        weights = np.concatenate(weights[channel_idxs], 
                                 weights[85 + channel_idxs], 
                                 weights[170 + channel_idxs])
        weights_tensor.set(weights.astype('float32'), place)
        # crop conv bias
        bias_tensor = fluid.global_scope().find_var(
                        "yolo_output.{}.conv.bias".format(i)).get_tensor()
        bias = np.array(bias_tensor)
        bias = np.concatenate(bias[channel_idxs], 
                              bias[85 + channel_idxs], 
                              bias[150 + channel_idxs])
        bias_tensor.set(bias.astype('float32'), place)

```

## FAQ

**Q:** 我使用单GPU训练，训练过程中`loss=nan`，这是为什么？  
**A:** YOLOv3中`learning_rate=0.001`的设置是针对总batch size为64的情况，若用户的batch size小于该值，建议调小学习率。

**Q:** 我训练YOLOv3速度比较慢，要怎么提速？  
**A:** YOLOv3的数据增强比较复杂，速度比较慢，可通过在[reader.py](./reader.py#L284)中增加数据读取的进程数来提速。若用户是进行fine-tune，也可将`--no_mixup_iter`设置大于`--max_iter`的值来禁用mixup提升速度。

**Q:** 我使用YOLOv3训练两个类别的数据集，训练`loss=nan`或推断结果不符合预期，这是为什么？  
**A:** `--label_smooth`参数会把所有正例的目标值设置为`1-1/class_num`，负例的目标值设为`1/class_num`，当`class_num`较小时，这个操作影响过大，可能会出现`loss=nan`或者训练结果错误，类别数较小时建议设置`--label_smooth=False`。若使用Paddle Fluid v1.5及以上版本，我们在C++代码中对这种情况作了保护，设置`--label_smooth=True`也不会出现这些问题。

## 参考文献

- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640v5), Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi.
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767v1), Joseph Redmon, Ali Farhadi.
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/abs/1902.04103v3), Zhi Zhang, Tong He, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li.

## 版本更新

- 1/2019, 新增YOLOv3模型。
- 4/2019, 新增YOLOv3模型Synchronized batch normalization模式。

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。

## 作者

- [heavengate](https://github.com/heavengate)
- [tink2123](https://github.com/tink2123)

