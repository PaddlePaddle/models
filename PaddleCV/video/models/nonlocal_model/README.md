# Non-local Neural Networks视频分类模型

---
## 目录

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型推断](#模型推断)
- [参考论文](#参考论文)


## 模型简介

Non-local Neural Networks是由Xiaolong Wang等研究者在2017年提出的模型，主要特点是通过引入关联函数来描述视频或者图像中像素点之间的非局域关联特性。在神经网络中，通常采用CNN或者RNN操作来增加空间或者时间邻域内的感受野，从而使得在神经网络的前向传播过程中，feature map上的像素点能够具有更多的全局信息。然而，经过一次CNN操作，输出feature map上的像素点，也只能获取其相应的感受野之内的信息，为了获得更多的上下文信息，就需要做多次卷积操作，以提升感受野的大小。Non-local模型引入了一种提取全局关联信息的方式，通过定义Nonlocal关联函数，输出feature map上的像素点，会跟输入feature map上的所有点相关联，能比CNN提取到更加全局的信息。

详细信息请参考论文[Non-local Neural Networks](https://arxiv.org/abs/1711.07971v1)

### Nonlocal操作

Nonlocal 关联函数的定义如下

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=y_{i}=\frac{1}{C(x)}&space;\sum_{j}f(x_i,&space;y_j)g(y_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{i}=\frac{1}{C(x)}&space;\sum_{j}f(x_i,&space;y_j)g(y_j)" title="y_{i}=\frac{1}{C(x)} \sum_{j}f(x_i, y_j)g(y_j)" /></a>
</p>

在上面的公式中，x表示输入feature map， y表示输出feature map，i是输出feature map的位置，j是输入feature map的位置，f(yi, xj)是输出点和输入点之间的关联性的函数，C是根据f(yi, xj)选取的归一化函数。g(xj)是对输入feature map做一个变换操作，通常可以选取比较简单的线性变换形式；f(yi, xj)可以选取不同的形式，通常可以选择

高斯式

内嵌高斯式

内积式

拼接式

### Nonlocal Block

采用类似Resnet的结构，定义如下的Nonlocal block

Nonlocal操作引入的部分与Resnet中的残差项类似，通过使用Nonlocal block，可以方便的在网络中的任何地方添加Nonlocal操作，而其他地方照样可以使用原始的预训练模型做初始化。如果将Wz初始化为0，则跟不使用Nonlocal block的初始情形等价。

### 具体实现

下图描述了使用内嵌高斯形式关联函数的具体实现过程，

其中XXXXXXX，g(Xj)是对输入feature map做一个线性变换，使用1x1x1的卷积；theta和phi也是线性变化，同样使用1x1x1的卷积来实现。二者的内积是
从上图中可以看到，Nonlocal操作只需用到通常的卷积、内积、加法、softmax等比较常用的算子，不需要额外添加新的算子，用户可以非常方便的实现组网构建模型。

### 模型效果

原作者的论文中指出，Nonlocal模型在视频分类问题上取得了较好的效果，在Resnet-50基础网络上添加Non-local block，能取得比Resnet-101更好的分类效果，TOP-1准确率要高出1～2个点。在图像分类和目标检测问题上，也有比较明显的提升效果。

## 数据准备

Non-local模型的训练数据采用由DeepMind公布的Kinetics-400动作识别数据集。数据下载及准备请参考Nonlocal模型的[数据说明](../../dataset/nonlocal/README.md)

## 模型训练

数据准备完毕后，可以通过如下两种方式启动训练：

    python train.py --model-name=NONLOCAL
            --config=./configs/nonlocal.txt
            --save-dir=checkpoints
            --log-interval=10
            --valid-interval=1

    bash scripts/train/train_nonlocal.sh

- 可下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/nonlocal_kinetics.tar.gz)通过`--resume`指定权重存放路径进行finetune等开发

**数据读取器说明：** 模型读取Kinetics-400数据集中的`mp4`数据，根据视频长度和采样频率随机选取起始帧的位置，每条数据抽取`sample_times`帧图像，对每帧图像做随机增强，短边缩放至[256, 320]之间的某个随机数，然后再crop出224x224的区域作为训练数据输入网络。

**训练策略：**

*  采用Momentum优化算法训练，momentum=0.9
*  采用L2正则化，卷积和fc层weight decay系数为1e-4；bn层则设置weight decay系数为0
*  base\_learning\_rate=0.01，在150,000和300,000的时候分别降一次学习率，衰减系数为0.1


## 模型评估

测试时数据预处理的方式跟训练时不一样，crop区域的大小为256x256，不同于训练时的224x224，所以需要将训练中预测输出时使用的全连接操作改为1x1x1的卷积。每个视频抽取图像帧数据的时候，会选取10个不同的位置作为时间起始点，做crop的时候会选取三个不同的空间起始点。在每个视频上会进行30次采样，将这30个样本的预测结果进行求和，选取概率最大的类别作为最终的预测结果。

可通过如下两种方式进行模型评估:

    python test.py --model-name=NONLOCAL
            --config=configs/nonlocal.txt
            --log-interval=1
            --weights=$PATH_TO_WEIGHTS

    bash scripts/test/test_nonlocal.sh

- 使用`scripts/test/test_nonlocal.sh`进行评估时，需要修改脚本中的`--weights`参数指定需要评估的权重。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/nonlocal_kinetics.tar.gz)进行评估


当取如下参数时:

| 参数 | 取值 |
| :---------: | :----: |
| back bone | Resnet-50 |
| 卷积形式 | c2d |
| 采样频率 | 8 |
| 视频长度 | 8 |

在Kinetics400的validation数据集下评估精度如下:

| 精度指标 | 模型精度 |
| :---------: | :----: |
| TOP\_1 | 0.739 |

### 备注

由于Youtube上部分数据已删除，只下载到了kinetics400数据集中的234619条，而原始数据集包含246535条视频，可能会导致精度略微下降。

## 模型推断

可通过如下命令进行模型推断：

    python infer.py --model_name=NONLOCAL
            --config=configs/nonlocal.txt
            --log_interval=1
            --weights=$PATH_TO_WEIGHTS
            --filelist=$FILELIST

- 模型推断结果存储于`NONLOCAL_infer_result`中，通过`pickle`格式存储。

- 若未指定`--weights`参数，脚本会下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/nonlocal_kinetics.tar.gz)进行推断


## 参考论文

- [Nonlocal Neural Networks](https://arxiv.org/abs/1811.01549), Dongliang He, Zhichao Zhou, Chuang Gan, Fu Li, Xiao Liu, Yandong Li, Limin Wang, Shilei Wen

