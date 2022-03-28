# PP-YOLOv2优化技巧

目标检测是计算机视觉领域中最常用的任务之一，除了零售商品以及行人车辆的检测，也有大量在工业生产中的应用：例如生产质检领域、设备巡检领域、厂房安全检测等领域。疫情期间，目标检测技术还被用于人脸口罩的检测，新冠肺炎检测等。同时目标检测技术整体流程较为复杂，对于不同的任务需要进行相应调整，因此目标检测模型的不断更新迭代拥有巨大的实用价值。

百度飞桨于2021年推出了业界顶尖的目标检测模型[PP-YOLOv2](https://arxiv.org/abs/2104.10419)，它是基于[YOLOv3](https://arxiv.org/abs/1804.02767)的优化模型，在尽可能不引入额外计算量的前提下提升模型精度。PP-YOLOv2(R50）在COCO 2017数据集mAP达到49.5%，在 640x640 的输入尺寸下，FPS 达到 68.9FPS，采用 TensorRT 加速，FPS 高达 106.5。PP-YOLOv2（R101）的mAP达到50.3%，对比当前最好的YOLOv5模型，相同的推理速度下，精度提升1.3%；相同精度下，推理速度加速15.9%。本章节重点围绕目标检测任务的优化技巧，并重点解读PP-YOLOv2模型的优化历程。

<div align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/232ff7ff4918482399c5f9276dc18bb4151918c4552143ca976e46cf10935a9b" width='800'/>
</div>


## 1. 目标检测算法优化思路


算法优化总体可以分为如下三部分，首先需要对目标场景进行详细分析，并充分掌握模型需求，例如模型体积、精度速度要求等。有效的前期分析有助于制定清晰合理的算法优化目标，并指导接下来高效的算法调研和迭代实验，避免出现尝试大量优化方法但是无法满足模型最终要求的情况。具体来说，调研和实验可以应用到数据模块、模型结构、训练策略三大模块。这个方法普遍的适用于深度学习模型优化，下面重点以目标检测领域为例，详细展开以上三部分的优化思路。
<div align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/25bfeaa9206e406aa0c406f0dcfabb85c20409b51f08401488f4007612d71fc6" width='800'/>
</div>

### 1.1 数据模块

数据模块可以说是深度学习领域最重要的一环，在产业应用中，数据往往都是自定义采集的，数据量相比开源数据集规模较小，因此高质量的标注数据和不断迭代是模型优化的一大利器。数据采集方面，少数精标数据的效果会优于大量粗标或者无标注数据，制定清晰明确的标注标准并覆盖尽可能全面的场景也是十分必要的。而在成本允许的情况下，数据是多多益善的。在学术研究中，通常是对固定的公开数据集进行迭代，此时数据模块的优化主要在数据增广方面，例如颜色、翻转、随机扩充、随机裁剪，以及近些年使用较多的[MixUp](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/image_augmentation/ImageAugment.html#mixup), [AutoAugment](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/image_augmentation/ImageAugment.html#autoaugment), Mosaic等方法。可以将不同的数据增广方法组合以提升模型泛化能力，需要注意的是，过多的数据增广可能会使模型学习能力降低，也使得数据加载模块耗时过长导致训练迭代效率降低；另外在目标检测任务中，部分数据增广方法可能会影响真实标注框的坐标位置，需要做出相应的调整。

<div align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/d153ee4cc94b4a4f8f2ebf86b76d18ab3b60ec6855ae4c4da2db66da1c4a79c5" width='800'/>
  <center><br>AutoAugment数据增广</br></center>
</div>


### 1.2 模型结构

模型结构方面存在一系列通用的优化方案，例如损失函数优化和特征提取优化，focal loss，IoU loss等损失函数优化能够在不影响推理速度的同时提升模型精度；[SPP](https://arxiv.org/abs/1406.4729)能够在几乎不增加预测耗时的情况下加强模型多尺度特征。

此外还需要在清晰的优化目标的基础上，作出针对性的优化。对于云端和边缘部署的场景，模型结构的设计选择不同，如下表所示。

| 场景      | 特点     | 模型结构建议 |
|:---------:|:------------------:|:------------:|
| 云端场景    | 算力充足，保证效果        | 倾向于使用ResNet系列骨干网络，并引入少量的可变形卷积实现引入少量计算量的同时提升模型特征提取能力。|
| 边缘端部署场景 | 算力和功耗相对云端较低，内存较小 | 倾向于使用[MobileNet](https://arxiv.org/abs/1704.04861)系列轻量级骨干网络，同时将模型中较为耗时的卷积替换为[深度可分离卷积](https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/Separable_Convolution.html?highlight=%E6%B7%B1%E5%BA%A6%E5%8F%AF%E5%88%86%E7%A6%BB%E5%8D%B7%E7%A7%AF#id4)，将[反卷积](https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/Transpose_Convolution.html?highlight=%E5%8F%8D%E5%8D%B7%E7%A7%AF)替换为插值的方式。 |


<div align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/86c783f67ecf439e9eab67f145bc7321b2828f68ef9f44efaff0e6be25a9985f" width='800'/>
</div>

除此之外，还有其他落地难点问题需要在模型结构中作出相应的调整及优化，例如小目标问题，可以选择大感受野的[HRNet](https://arxiv.org/pdf/1904.04514.pdf), [DLA](https://arxiv.org/pdf/1707.06484.pdf)等网络结构作为骨干网络；并将普通卷积替换为[空洞卷积](https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/Dilated_Convolution.html?highlight=%E7%A9%BA%E6%B4%9E%E5%8D%B7%E7%A7%AF)。对于数据长尾问题，可以在损失函数部分抑制样本较多的类别，提升样本较少类别的重要性。

### 1.3 训练策略

在模型迭代优化的过程中，会引入不同的优化模块，可能导致模型训练不稳定，为此需要改进训练策略，加强模型训练稳定性，同时提升模型收敛效果。所有训练策略的调整并不会对预测速度造成损失。例如调整优化器、学习率等训练参数，Synchronized batch normalization(卡间同步批归一化)能够扩充batch信息，使得网络获取更多输入信息，EMA（Exponential Moving Average）通过滑动平均的方式更新参数，避免异常值对参数的影响。在实际应用场景中，由于数据量有限，可以使用预先在COCO数据集上训练好的模型进行迁移学习，能够大幅提升模型精度。该模块中的优化策略也能够通用的提升不同计算机视觉任务模型效果。

以上优化技巧集成了多种算法结构及优化模块，需要大量的代码开发，而且不同优化技巧之间需要相互组合，对代码的模块化设计有较大挑战，接下来介绍飞桨推出的一套端到端目标检测开发套件PaddleDetection。

下面结合代码具体讲解如何使用PaddleDetection将YOLOv3模型一步步优化成为业界SOTA的PP-YOLOv2模型

## 2. PP-YOLO优化及代码实践

YOLO系列模型一直以其高性价比保持着很高的使用率和关注度，近年来关于YOLO系列模型上的优化和拓展的研究越来越多，其中有[YOLOv4](https://arxiv.org/abs/2004.10934),YOLOv5,旷视发布的[YOLOX](https://arxiv.org/abs/2107.08430)系列模型，它们整合了计算机视觉的state-of-the-art技巧，大幅提升了YOLO目标检测性能。百度飞桨通过自研的目标检测框架PaddleDetection，对YOLOv3进行细致优化，在尽可能不引入额外计算量的前提下提升模型精度，今年推出了高精度低时延的PP-YOLOv2模型。下面分别从数据增广、骨干网络、Neck&head结构、损失函数、后处理优化、训练策略几个维度详细展开。

### 2.1 数据增广

PP-YOLOv2中采用了大量数据增广方式，这里逐一进行说明

#### 2.1.1 MixUp

[MixUp](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/data/transform/operators.py#L1574)以随机权重对图片和标签进行线性插值，在目标检测任务中标签向量gt_bbox，gt_class，is_crowd等直接连接，gt_score进行加权求和。Mixup可以提高网络在空间上的抗干扰能力，线性插值的权重满足Beta分布，表达式如下：
$$
\widetilde x = \lambda x_i + (1 - \lambda)x_j,\\
\widetilde y = \lambda y_i + (1 - \lambda)y_j \\
\lambda\in[0,1]
$$
以下图为例，将任意两张图片加权叠加作为输入。
<div align=center>
    <img src="https://raw.githubusercontent.com/mls1999725/pictures/master/Mixup.png" alt="Mixup" style="zoom: 80%;"/>
</div>


#### 2.1.2 RandomDistort

[RandomDistort](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/data/transform/operators.py#L329)操作以一定的概率对图像进行随机像素内容变换，包括色相（hue），饱和度（saturation），对比度（contrast），明亮度（brightness）。


#### 2.1.3 RandomExpand

[随机扩展](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/data/transform/operators.py#L875)（RandomExpand）图像的操作步骤如下：

- 随机选取扩张比例（扩张比例大于1时才进行扩张）。
- 计算扩张后图像大小。
- 初始化像素值为输入填充值的图像，并将原图像随机粘贴于该图像上。
- 根据原图像粘贴位置换算出扩张后真实标注框的位置坐标。


#### 2.1.4 RandomCrop

[随机裁剪](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/data/transform/operators.py#L1182)（RandomCrop）图像的操作步骤如下：

- 若allow_no_crop为True，则在thresholds加入’no_crop’。
- 随机打乱thresholds。
- 遍历thresholds中各元素： (1) 如果当前thresh为’no_crop’，则返回原始图像和标注信息。 (2) 随机取出aspect_ratio和scaling中的值并由此计算出候选裁剪区域的高、宽、起始点。 (3) 计算真实标注框与候选裁剪区域IoU，若全部真实标注框的IoU都小于thresh，则继续第（3）步。 (4) 如果cover_all_box为True且存在真实标注框的IoU小于thresh，则继续第（3）步。 (5) 筛选出位于候选裁剪区域内的真实标注框，若有效框的个数为0，则继续第（3）步，否则进行第（4）步。
- 换算有效真值标注框相对候选裁剪区域的位置坐标。


#### 2.1.5 RandomFlip

[随机翻转](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/data/transform/operators.py#L487)（RandomFlip）操作利用随机值决定是否对图像，真实标注框位置进行翻转。



以上数据增广方式均在[ppyolov2_reader.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/_base_/ppyolov2_reader.yml#L5)进行配置


### 2.2 骨干网络

不同于YOLOv3的DarkNet53骨干网络，PP-YOLOv2使用更加优异的ResNet50vd-DCN作为模型的骨干网络。它可以被分为ResNet50vd和DCN两部分来看。ResNet50vd是指拥有50个卷积层的ResNet-D网络。ResNet结构如下图所示：

<div align=center>
    <img src="https://raw.githubusercontent.com/mls1999725/pictures/master/ResNet-A.png" alt="ResNet-A" style="zoom: 50%;"/>
</div>

ResNet系列模型在2015年提出后，其模型结构不断被业界开发者持续改进，在经过了B、C、D三个版本的改进后，最新的ResNetvd结构能在基本不增加计算量的情况下显著提高模型精度。ResNetvd的第一个卷积层由三个卷积构成，卷积核尺寸均是3x3，步长分别为2，1，1，取代了上图的7x7卷积，在参数量基本不变的情况下增加网络深度。同时，ResNet-D在ResNet-B的基础上，在下采样模块加入了步长为2的2x2平均池化层，并将之后的卷积步长修改为1，避免了输入信息被忽略的情况。B、C、D三种结构的演化如下图所示：

<div align=center>
    <img src="https://raw.githubusercontent.com/mls1999725/pictures/master/resnet结构.png" alt="resnet结构" style="zoom: 50%;"/>
</div>

ResNetvd下采样模块代码参考实现：[代码链接](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/modeling/backbones/resnet.py#L265)

ResNetvd使用方式参考[ResNetvd配置](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/_base_/ppyolov2_r50vd_dcn.yml#L13)

```
ResNet:
  depth: 50              # ResNet网络深度
  variant: d             # ResNet变种结构，d即表示ResNetvd
  return_idx: [1, 2, 3]  # 骨干网络引出feature map层级
  dcn_v2_stages: [3]     # 引入可变形卷积层级
  freeze_at: -1          # 不更新参数的层级
  freeze_norm: false     # 是否不更新归一化层
  norm_decay: 0.         # 归一化层对应的正则化系数
```

经多次实验发现，使用ResNet50vd结构作为骨干网络，相比于原始的ResNet，可以提高1%-2%的目标检测精度，且推理速度基本保持不变。而DCN（Deformable Convolution）可变形卷积的特点在于：其卷积核在每一个元素上额外增加了一个可学习的偏移参数。这样的卷积核在学习过程中可以调整卷积的感受野，从而能够更好的提取图像特征，以达到提升目标检测精度的目的。但它会在一定程度上引入额外的计算开销。经过多翻尝试，发现只在ResNet的最后一个stage增加可变形卷积，是实现引入极少计算量并提升模型精度的最佳策略。

可变形卷积的[代码实现](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/modeling/layers.py#L41)如下：

```python
from paddle.vision.ops import DeformConv2D

class DeformableConvV2(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 regularizer=None,
                 skip_quant=False,
                 dcn_bias_regularizer=L2Decay(0.),
                 dcn_bias_lr_scale=2.):
        super().__init__()
        self.offset_channel = 2 * kernel_size**2
        self.mask_channel = kernel_size**2

        offset_bias_attr = ParamAttr(
            initializer=Constant(0.),
            learning_rate=lr_scale,
            regularizer=regularizer)

        self.conv_offset = nn.Conv2D(
            in_channels,
            3 * kernel_size**2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=ParamAttr(initializer=Constant(0.0)),
            bias_attr=offset_bias_attr)

        if bias_attr:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            dcn_bias_attr = ParamAttr(
                initializer=Constant(value=0),
                regularizer=dcn_bias_regularizer,
                learning_rate=dcn_bias_lr_scale)
        else:
            # in ResNet backbone, do not need bias
            dcn_bias_attr = False
        self.conv_dcn = DeformConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=dcn_bias_attr)

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y
```


### 2.3 Neck&head结构

PP-YOLOv2模型中使用PAN和SPP结构来强化模型结构的Neck部分。[PAN（Path Aggregation Network）](https://arxiv.org/abs/1803.01534)结构，作为[FPN](https://arxiv.org/abs/1612.03144)的变形之一，通过从上至下和从下到上两条路径来聚合特征信息，达到更好的特征提取效果。具体结构如下图，其中C3, C4, C5为3个不同level的feature，分别对应stride为(8, 16, 32)；其中Detection Block使用CSP connection方式，对应ppdet的[PPYOLODetBlockCSP模块](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/modeling/necks/yolo_fpn.py#L359)

<div align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/eeae465462484a6a9797f779434ef721cb9882eb10374b34be43956360691521" width='600'/>
</div>

SPP在[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf)中提出，可以通过多个不同尺度的池化窗口提取不同尺度的池化特征，然后把特征组合在一起作为输出特征，能有效的增加特征的感受野，是一种广泛应用的特征提取优化方法。PPYOLO-v2中使用三个池化窗口分别是(5, 9, 13)，得到特征通过concat拼接到一起，最后跟一个卷积操作，详见[SPP模快](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/modeling/necks/yolo_fpn.py#L114)。SPP会插入到PAN第一组计算的[中间位置](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/modeling/necks/yolo_fpn.py#L903)。

<div align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/439a29465efc4867ac4edc70d17d3ac9aa124d719b364170b08422a685045745" width='600'/>
</div>

除此之外，PP-YOLOv2 Neck部分引入了[Mish](https://arxiv.org/pdf/1908.08681.pdf)激活函数，公式如下：

$$
mish(x) = x \ast tanh(ln(1+e^x))
$$

Mish的[代码实现](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/modeling/ops.py#L43)如下所示：

```python
def mish(x):
    return x * paddle.tanh(F.softplus(x))
```


PP-YOLOv2中PAN模块使用方式参考 [neck: PPYOLOPAN](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/_base_/ppyolov2_r50vd_dcn.yml#L9)

```
PPYOLOPAN:
  act: "mish"        # 默认使用mish函数
  conv_block_num: 2  # 每个pan block中使用的conv block个数
  drop_block: true   # 是否采用drop block, 训练策略模块中介绍
  block_size: 3      # DropBlock的size
  keep_prob: 0.9     # DropBlock保留的概率
  spp: true          # 是否使用spp
```

PP-YOLOv2的Head部分在PAN输出的3个scale的feature上进行预测，PP-YOLOv2采用和[YOLO-v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)类似的结构，即使用卷积对最后的feature进行编码，最后输出的feature是四维的tensor，分别是[n, c, h, w]对应图像数量、通道数、高和宽。c是具体的形式为anchor_num ∗ (4 + 1 + 1 + num_classs)，anchor_num是每个位置对应的anchor的数量(PP-YOLOv2中为3)，4代表bbox的属性(对应中心点和长宽)，1代表是否是物体(objectness), 1代表iou_aware(详细见损失函数计算), num_classs代表类别数量(coco数据集上为80).

使用方式参考[yolo_head: YOLOv3Head](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/_base_/ppyolov2_r50vd_dcn.yml#L28)

```
YOLOv3Head:
  # anchors包含9种, 根据anchor_masks的index分为3组，分别对应到不同的scale
  # [6, 7, 8]对应到stride为32的预测特征
  # [3, 4, 5]对应到stride为16的预测特征
  # [0, 1, 2]对应到stride为8的预测特征
  anchors: [[10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]]
  anchor_masks: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
  loss: YOLOv3Loss      # 采用损失函数类型，详细见损失函数模块
  iou_aware: true       # 是否使用iou_aware
  iou_aware_factor: 0.5 # iou_aware的系数
```


### 2.4 损失函数

PP-YOLOv2使用IoU Loss和IoU Aware Loss提升定位精度。IoU Loss直接优化预测框与真实框的IoU，提升了预测框的质量。IoU Aware Loss则用于监督模型学习预测框与真实框的IoU，学习到的IoU将作为定位置信度参与到NMS的计算当中。

对于目标检测任务，IoU是我们常用评估指标。预测框与真实框的IoU越大，预测框与真实框越接近，预测框的质量越高。基于“所见即所得”的思想，PP-YOLOv2使用IoU Loss直接去优化模型的预测框与真实框的IoU。IoU Loss的表达式如下：

$$
L_{iou}=1 - iou^2
$$

IoU Loss的[代码实现](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/modeling/losses/iou_loss.py#L56)如下所示：

```python
iou = bbox_iou(
    pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
if self.loss_square:
    loss_iou = 1 - iou * iou
else:
    loss_iou = 1 - iou

loss_iou = loss_iou * self.loss_weight
```

PP-YOLOv2增加了一个通道用于学习预测框与真实框的IoU，并使用IoU Aware Loss来监督这一过程。在推理过程中，将这个通道学习的IoU预测值也作为评分的因子之一，能一定程度上避免高IoU预测框被挤掉的情况，从而提升模型的精度。IoU Aware Loss为二分类交叉熵损失函数，其表达式如下：

$$
L_{iou\_aware} = -(iou * log(ioup) + (1 - iou) * log(1 - ioup))
$$

IoU Aware Loss的[代码实现](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/modeling/losses/iou_aware_loss.py#L41)如下：

```python
iou = bbox_iou(
    pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
iou.stop_gradient = True
loss_iou_aware = F.binary_cross_entropy_with_logits(
    ioup, iou, reduction='none')
loss_iou_aware = loss_iou_aware * self.loss_weight
```

### 2.5 后处理优化

在后处理的过程中，PP-YOLOv2采用了Matrix NMS和Grid Sensitive。Matrix NMS为并行化的计算[Soft NMS](https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/object_detection/SoftNMS.html?highlight=Soft%20NMS)的算法，Grid Sensitive解决了检测框的中心落到网格边线的情况。

Grid Sensitive是YOLOv4引入的优化方法，如下图所示，YOLO系列模型中使用sigmoid函数来预测中心点相对于grid左上角点的偏移量。

<div align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/2c34ba7dc29b41feb09455c71ffe444f0d06c733b7384ec7bf8f56357fd6375c", width='400'/>
</div>

然而，当中心点位于grid的边线上时，使用sigmoid函数较难预测。因此，对于预测值加上一个缩放和偏移，保证预测框中心点能够有效的拟合真实框刚好落在网格边线上的情况。Grid Sensitive的表达式如下：

$$
x = scale * \sigma(x) - 0.5 * (scale - 1.) \\
y = scale * \sigma(y) - 0.5 * (scale - 1.)
$$

Matrix NMS通过一个矩阵并行运算的方式计算出任意两个框之间的IoU，从而实现并行化的计算Soft NMS，在提升检测精度的同时，避免了推理速度的下降。Matrix NMS的实现在PaddlePaddle框架的[Matrix NMS OP](https://github.com/PaddlePaddle/Paddle/blob/release/2.1/paddle/fluid/operators/detection/matrix_nms_op.cc#L169)中，在PaddleDetection中封装了[Matrix NMS API](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/modeling/layers.py#L426)

使用方式参考：[post process: MatrixNMS](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/_base_/ppyolov2_r50vd_dcn.yml#L59)
```
nms:
    name: MatrixNMS       # NMS类型，支持MultiClass NMS和Matrix NMS
    keep_top_k: 100       # NMS输出框的最大个数
    score_threshold: 0.01 # NMS计算前的分数阈值
    post_threshold: 0.01  # NMS计算后的分数阈值
    nms_top_k: -1         # NMS计算前，分数过滤后保留的最大个数
    background_label: -1  # 背景类别
```

### 2.6 训练策略

在训练过程中，PP-YOLOv2使用了Synchronize batch normalization, EMA(Exponential Moving Average，指数滑动平均)和DropBlock和来提升模型的收敛效果以及泛化性能。

BN(Batch Normalization, 批归一化)是训练卷积神经网络时常用的归一化方法，能起到加快模型收敛，防止梯度弥散的效果。在BN的计算过程中，需要统计样本的均值和方差，通常batch size越大，统计得到的均值和方差越准确。在多卡训练时，样本被等分送入每张卡，如果使用BN进行归一化，每张卡会利用自身的样本分别计算一个均值和方差进行批处理化，而SyncBN会同步所有卡的样本信息统一计算一个均值和方差，每张卡利用这个均值和方差进行批处理化。因此，使用SyncBN替代BN，能够使计算得到的均值和方差更加准确，从而提升模型的性能。SyncBN的[代码实现](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/modeling/backbones/resnet.py#L104)如下所示：

```python
if norm_type == 'sync_bn':
    self.norm = nn.SyncBatchNorm(
        ch_out, weight_attr=param_attr, bias_attr=bias_attr)
else:
    self.norm = nn.BatchNorm(
        ch_out,
        act=None,
        param_attr=param_attr,
        bias_attr=bias_attr,
        use_global_stats=global_stats)
```

使用方法参考：[SyncBN](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/_base_/ppyolov2_r50vd_dcn.yml#L3)

```
norm_type: sync_bn
```

EMA是指将参数过去一段时间的均值作为新的参数。相比直接对参数进行更新，采用滑动平均的方式能让参数学习过程中变得更加平缓，能有效避免异常值对参数更新的影响，提升模型训练的收敛效果。EMA包含更新和校正两个过程，更新过程使用指数滑动平均的方式不断地更新参数$\theta$, 校正过程通过除以$(1 - decay^t)$来校正对于初值的偏移。

EMA的更新过程如下列表达式所示：

$$
\theta_0 = 0 \\
\theta_t = decay * \theta_{t - 1} + (1 - decay) * \theta_t
$$

EMA的校正过程如下列表达式所示:

$$
\tilde{\theta_t} = \frac{\theta_t}{1 - decay^t}
$$

EMA的[代码实现](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/optimizer.py#L261)如下所示：

```python
def update(self, model):
    if self.use_thres_step:
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
    else:
        decay = self.decay
    self._decay = decay
    model_dict = model.state_dict()
    for k, v in self.state_dict.items():
        v = decay * v + (1 - decay) * model_dict[k]
        v.stop_gradient = True
        self.state_dict[k] = v
    self.step += 1

def apply(self):
    if self.step == 0:
        return self.state_dict
    state_dict = dict()
    for k, v in self.state_dict.items():
        v = v / (1 - self._decay**self.step)
        v.stop_gradient = True
        state_dict[k] = v
    return state_dict
```

使用方式参考：[EMA](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/_base_/ppyolov2_r50vd_dcn.yml#L4)
```
use_ema: true
ema_decay: 0.9998
```

与[Dropout](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/model_tuning/regularization/dropout.html?highlight=Dropout)类似，DropBlock是一种防止过拟合的方法。因为卷积特征图的相邻点之间包含密切相关的语义信息，以特征点的形式随机Drop对于目标检测任务通常不太有效。基于此，DropBlock算法在Drop特征的时候不是以特征点的形式来Drop的，而是会集中Drop掉某一块区域，从而更适合被应用到目标检测任务中来提高网络的泛化能力，如下图(c)中所示。

<div align="center">
  <img src="https://ai-studio-static-online.cdn.bcebos.com/cf257e09a8164b19bf0e6adc0eabbce0123917146c624e90a0e10f68ea38bb4b", width='600'/>
</div>


DropBlock的[代码实现](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/ppdet/modeling/necks/yolo_fpn.py#L196)如下所示：

```python
gamma = (1. - self.keep_prob) / (self.block_size**2)
if self.data_format == 'NCHW':
    shape = x.shape[2:]
else:
    shape = x.shape[1:3]
for s in shape:
    gamma *= s / (s - self.block_size + 1)

matrix = paddle.cast(paddle.rand(x.shape, x.dtype) < gamma, x.dtype)
mask_inv = F.max_pool2d(
            matrix,
            self.block_size,
            stride=1,
            padding=self.block_size // 2,
            data_format=self.data_format)
mask = 1. - mask_inv
y = x * mask * (mask.numel() / mask.sum())
```


以上是PP-YOLOv2模型优化的全部技巧，期间也实验过大量没有正向效果的方法，这些方法可能并不适用于YOLO系列的模型结构或者训练策略，在[PP-YOLOv2](https://arxiv.org/abs/2104.10419)论文中汇总了一部分，这里不详细展开了。下面分享PP-YOLOv2在实际应用中的使用技巧和模型调优经验。


## 3. 调参经验

### 3.1 配置合理的学习率

PaddleDetection提供的[学习率配置](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/_base_/optimizer_365e.yml)是使用8张GPU训练，每张卡batch_size为12时对应的学习率（base_lr=0.005）,如果在实际训练时使用了其他的GPU卡数或batch_size，需要相应调整学习率设置，否则可能会出现模型训练出nan的情况。调整方法为的学习率与总batch_size，即卡数乘以每张卡batch_size，成正比，下表举例进行说明


| GPU卡数 | 每张卡batch_size | 总batch_size | 对应学习率 |
| -------- | -------- | -------- | -------- |
| 8     | 12     | 96     |   0.005       |
| 1     | 12     | 12     |   0.000625    |
| 8     | 6     | 48     |   0.0025       |


### 3.2  在资源允许的情况下增大batch_size。

在多个目标检测任务优化的过程发现，仅仅增大reader中的batch_size有助于提升模型收敛效果。

### 3.3 调整gradient clip

在PP-YOLOv2中，设置了[clip_grad_by_norm](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/_base_/optimizer_365e.yml#L15) 为35以防止模型训练梯度爆炸，对于自定义任务，如果出现了梯度爆炸可以尝试修改梯度裁剪的值。
