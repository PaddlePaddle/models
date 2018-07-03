运行本目录下的程序示例需要使用 PaddlePaddle 最新的 develop branch 版本。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_cn.html)中的说明更新 PaddlePaddle 安装版本。

---


## Pyramidbox 人脸检测

## Table of Contents
- [简介](#简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型发布](#模型发布)

### 简介

在不受控制的环境中，检测小的、模糊的和部分遮挡的人脸是一个挑战。[PyramidBox](https://arxiv.org/pdf/1803.07737.pdf) 是一种基于SSD的单阶段人脸检测器，它利用上下文信息解决困难人脸的检测问题。如下图所示，PyramidBox在六个尺度的特征图上进行不同层级的预测。该工作主要包括以下模块：LFPN、Pyramid Anchors、CPM、Data-anchor-sampling。具体可以参考该方法对应的论文 https://arxiv.org/pdf/1803.07737.pdf ，下面进行简要的介绍。
<p align="center">
<img src="images/architecture_of_pyramidbox.jpg" height=300 width=900 hspace='10'/> <br />
Pyramidbox 人脸检测模型
</p>

**LFPN**: LFPN全称Low-level Feature Pyramid Networks, 在检测任务中，LFPN可以充分结合高层次的包含更多上下文的特征和低层次的包含更多纹理的特征。高层级特征被用于检测尺寸较大的人脸，而低层级特征被用于检测尺寸较小的人脸。为了将高层级特征整合到高分辨率的低层级特征上，我们从中间层开始做自上而下的融合，构建Low-level FPN。另外，该中间层的感受野接近于输入尺寸的一半。

**Pyramid Anchors**: 该算法使用半监督解决方案来生成与人脸检测相关的具有语义的近似标签，提出基于anchor的语境辅助方法，它引入有监督的信息来学习较小的、模糊的和部分遮挡的人脸的语境特征。使用者可以根据ground truth的人脸标签，按照一定的比例扩充，得到头部的标签（上下左右各扩充1/2）和人体的标签（可自定义扩充比例）。

**CPM**: CPM全称Context-sensitive Predict Module, 本方法设计了一种上下文敏感结构(CPM)来提高预测网络的表达能力。

**Data-anchor-sampling**: 设计了一种新的采样方法，称作Data-anchor-sampling，该方法可以增加训练样本在不同尺度上的多样性。该方法改变训练样本的分布，重点关注较小的人脸。

Pyramidbox模型可以在以下示例图片上展示鲁棒的检测性能，该图有一千张人脸，该模型检测出其中的880张人脸。
<p align="center">
<img src="images/demo_img.jpg" height=300 width=900 hspace='10'/> <br />
Pyramidbox 人脸检测性能展示
</p>



### 数据准备

你可以使用 [WIDER FACE 数据集](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) 来进行模型的训练测试工作，官网给出了详尽的数据介绍。

WIDER FACE数据集包含32,203张图片，其中包含393,703个人脸，数据集的人脸在尺度、姿态、遮挡方面有较大的差异性。另外WIDER FACE数据集是基于61个事件类归类的，然后针对每个事件类，随意的挑选40%作为训练集，10%作为验证集，50%作为测试集。

从官网训练集和验证集，放在`data`目录，官网提供了谷歌云和百度云下载地址，请依据情况自行下载。并下载训练集和验证集的标注信息:

```bash
./data/download.sh
```

准备好数据之后，`data`目录如下：

```
data
|-- download.sh
|-- wider_face_split
|   |-- readme.txt
|   |-- wider_face_train_bbx_gt.txt
|   |-- wider_face_val_bbx_gt.txt
|   `-- ...
|-- WIDER_train
|   `-- images
|       |-- 0--Parade
|       ...
|       `-- 9--Press_Conference
`-- WIDER_val
    `-- images
        |-- 0--Parade
        ...
        `-- 9--Press_Conference
```


### 模型训练

#### 下载预训练模型

我们提供了预训练模型，模型基于VGGNet主干网络训练。


```bash
wget http://paddlemodels.bj.bcebos.com/vgg_ilsvrc_16_fc_reduced.tar.gz
tar -xf vgg_ilsvrc_16_fc_reduced.tar.gz && rm -f vgg_ilsvrc_16_fc_reduced.tar.gz
```

声明：该预训练模型转换自[Caffe](http://cs.unc.edu/~wliu/projects/ParseNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel)，我们不久也会发布我们自己预训练的模型。


#### 开始训练


`train.py` 是训练模块的主要执行程序，调用示例如下：

```bash
python -u train.py --batch_size=16 --pretrained_model=vgg_ilsvrc_16_fc_reduced
```
  - 可以通过设置 `export CUDA_VISIBLE_DEVICES=0,1,2,3` 指定想要使用的GPU数量。
  - 更多的可选参数见:
    ```bash
    python train.py --help
    ```

模型训练所采用的数据增强：

**数据增强**：数据的读取行为定义在 `reader.py` 中，所有的图片都会被缩放到640x640。在训练时还会对图片进行数据增强，包括随机扰动、扩张、翻转、裁剪，和[物体检测](https://github.com/PaddlePaddle/models/blob/develop/fluid/object_detection/README_cn.md#%E8%AE%AD%E7%BB%83-pascal-voc-%E6%95%B0%E6%8D%AE%E9%9B%86)中数据增强类似，除此之外，增加了上面提到的Data-anchor-sampling:

  **尺度变换(Data-anchor-sampling)**：根据SSD模型中anchor的配置来随机将图片尺度变换到一定范围的尺度，大大增强人脸的尺度变化。具体操作为根据随机选择的人脸长heigh和宽width，得到$val=\sqrt{width*heigh}$，判断$val$的值在表示scale相关的向量的哪个区间[16，32，64，128，256，512]。假设val=45，则选定32<val<64，以均匀分布的概率选取[16，32，64]中的任意一个值。若选中64，则该人脸的resize区间在[64/2，min(val*2, 64*2)]中选定。



**注意**：
  - 本次开源模型中CPM模块与论文中有些许不同，相比论文中CPM模块训练和测试速度更快。
  - Pyramid Anchors模块的body部分可以针对不同情况，进行相应的长宽设置来调参。同时face、head、body部分的loss对应的系数也可以通过调参优化。


### 模型评估

验证集的评估需要两个步骤：先预测出验证集的检测框和置信度，再利用WIDER FACE官方提供的评估脚本得到评估结果。

1. 预测检测结果

```bash
python -u widerface_eval.py --model_dir=output/159 --save_dir=pred
```

更多的可选参数:

```bash
python -u widerface_eval.py --help
```


- 评估优化： `widerface_eval.py`中multi_scale_test_pyramid可用可不用，由于Data-anchor-sampling的作用，更加密集的anchors对性能有更大的提升。

2. 评估指标

下载官方评估脚本:

```
wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip
unzip eval_tools.zip && rm -f eval_tools.zip
```

修改`eval_tools/wider_eval.m`中检测结果保存的路径和将要画出的曲线名称：

```
# 此处修改存放结果的文件夹名字
pred_dir = './pred';  
# 此处修改将要画出的曲线名称
legend_name = 'Fluid-PyramidBox';
```

`wider_eval.m`是评估模块的主要执行程序，命令行式的运行命令如下：

```bash
matlab -nodesktop -nosplash -nojvm -r "run wider_eval.m;quit;"
```


### 模型发布



| 模型                    | 预训练模型  | 训练数据    | 测试数据    | mAP |
|:------------------------:|:------------------:|:----------------:|:------------:|:----:|
|[Pyramidbox-v1-SSD 640x640]() | [VGGNet](http://paddlemodels.bj.bcebos.com/vgg_ilsvrc_16_fc_reduced.tar.gz) | WIDER FACE train | WIDER FACE Val   | 95.6%/ 94.6%/ 89.4%  |
