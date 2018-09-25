# Faster RCNN 目标检测

---
## 内容

- [安装](#安装)
- [简介](#简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [参数微调](#参数微调)
- [模型评估](#模型评估)
- [模型推断及可视化](#模型推断及可视化)
- [附录](#附录)

## 安装

在当前目录下运行样例代码需要PadddlePaddle Fluid的v0.13.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://www.paddlepaddle.org/documentation/docs/zh/0.15.0/beginners_guide/install/install_doc.html#paddlepaddle)中的说明来更新PaddlePaddle。

## 简介

[Faster Rcnn](https://arxiv.org/abs/1506.01497) 是典型的两阶段目标检测器。如下图所示，整体网络可以分为4个主要内容：
<p align="center">
<img src="image/Faster_RCNN.jpg" height=400 width=400 hspace='10'/> <br />
Faster RCNN 目标检测模型
</p>

1. 基础卷积层。作为一种卷积神经网络目标检测方法，Faster RCNN首先使用一组基础的卷积网络提取图像的feature maps。feature maps被后续RPN层和全连接层共享。本示例采用[ResNet-50](https://arxiv.org/abs/1512.03385)作为基础卷积层。
2. 区域生成网络(RPN)。RPN网络用于生成候选区域(proposals)。该层通过一组固定的尺寸和比例得到一组锚点(anchors), 通过softmax判断anchors属于前景或者背景，在利用box regression修正anchors获得精确的proposals。
3. RoI池化。该层收集输入的feature maps和proposals，将proposals映射到feature maps中并池化为统一大小的proposal feature maps，送入全连接层判定目标类别。
4. 检测层。利用proposal feature maps计算proposal的类别，同时再次box regression获得检测框最终的精确位置。

## 数据准备

在[MS-COCO数据集](http://cocodataset.org/#download)上进行训练，通过如下方式下载数据集。

    cd dataset/coco
    ./download.sh

## 模型训练

数据准备完毕后，可以通过如下的方式启动训练：

    python train.py \
       --max_size=1333 \
       --scales=800 \
       --batch_size=8 \
       --batch_size_per_im=512 \
       --class_dim=81 \
       --model_save_dir=output/ \
       --max_iter=180000 \
       --learning_rate=0.01 \
       --padding_minibatch=True

- 通过设置export CUDA\_VISIBLE\_DEVICES=0,1指定GPU训练。
- 可选参数见：

    python train.py --help

**数据读取器说明：** 数据读取器定义在reader.py中。所有图像将短边等比例缩放至`scales`，若长边大于`max_size`, 则再次将长边等比例缩放至`max_iter`。在训练阶段，对图像采用水平翻转。支持将同一个batch内的图像padding为相同尺寸。

**模型设置：**

* roi_pool为average pooling。
* 训练过程pre\_nms=12000, post\_nms=2000，测试过程pre\_nms=6000, post\_nms=1000。nms阈值为0.7。
* RPN网络得到labels的过程中，fg\_fraction=0.25，fg\_thresh=0.5，bg\_thresh_hi=0.5，bg\_thresh\_lo=0.0
* RPN选择anchor时，rpn\_fg\_fraction=0.5，rpn\_positive\_overlap=0.7，rpn\_negative\_overlap=0.3


下图为模型训练结果：
<p align="center">
<img src="image/train_loss.jpg" height=500 width=650 hspace='10'/> <br />
Faster RCNN 训练loss
</p>

**训练策略：**

*  采用momentum优化算法训练Faster RCNN，momentum=0.9。
*  权重衰减系数为0.0001，前500轮学习率从0.00333线性增加至0.01。在120000，160000轮时使用0.1,0.01乘子进行学习率衰减，最大训练180000轮。
*  非基础卷积层卷积bias学习率为整体学习率2倍。
*  基础卷积层中，affine_layers参数不更新，res2层参数不更新。
*  使用Nvidia Tesla V100 8卡并行，总共训练时长大约40小时。

## 参数微调

参数微调是指在特定任务上微调已训练模型的参数。通过初始化`pretrained_model`，微调一个模型可以采用如下的命令：

    python train.py
        --max_size=1333 \
        --scales=800 \
        --pretrained_model=${path_to_pretrain_model} \
        --batch_size= 8\
        --model_save_dir=output/ \
        --class_dim=81 \
        --max_iter=180000 \
        --learning_rate=0.01 \

## 模型评估

模型评估是指对训练完毕的模型评估各类性能指标。本示例采用[COCO官方评估](http://cocodataset.org/#detections-eval)，使用前需要首先下载[cocoapi](https://github.com/cocodataset/cocoapi)：

    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    # if cython is not installed
    pip install Cython
    # Install into global site-packages
    make install
    # Alternatively, if you do not have permissions or prefer
    # not to install the COCO API into global site-packages
    python2 setup.py install --user

`eval_coco_map.py`是评估模块的主要执行程序，调用示例如下：

    python eval_coco_map.py \
        --dataset=coco2017 \
        --pretrained_mode=${path_to_pretrain_model} \
        --batch_size=1 \
        --nms_threshold=0.5 \
        --score_threshold=0.05

下图为模型评估结果：
<p align="center">
<img src="image/mAP.jpg" height=500 width=650 hspace='10'/> <br />
Faster RCNN mAP
</p>

## 模型推断及可视化

模型推断可以获取图像中的物体及其对应的类别，`infer.py`是主要执行程序，调用示例如下：

    python infer.py \
       --dataset=coco2017 \
        --pretrained_model=${path_to_pretrain_model}  \
        --image_path=data/COCO17/val2017/  \
        --image_name=000000000139.jpg \
        --draw_threshold=0.6

下图为模型可视化预测结果：
<p align="center">
<img src="image/000000000139.jpg" height=300 width=400 hspace='10'/>
<img src="image/000000127517.jpg" height=300 width=400 hspace='10'/>
<img src="image/000000203864.jpg" height=300 width=400 hspace='10'/>
<img src="image/000000515077.jpg" height=300 width=400 hspace='10'/> <br />
Faster RCNN 预测可视化
</p>

## 附录

### 调试模型

由于Faster RCNN所需参数较多，模型较为复杂，常需要对各个模块进行调试。

#### 输入

Fluid采用如下方法对模块输入进行赋值：

    rpn_rois_t = np.load('rpn_rois')
    rpn_rois = fluid.core.LoDTensor()
    rpn_rois.set(rpn_rois_t, place)

如果输入variable为lodTensor，则需另外设置lod：

    rpn_rois.set_lod(lod)

在调试过程中，需要将`use_random`设置为False

#### 输出

调试过程中，需要找到待调试变量的名字，提供如下两种方法：

1. 通过program.global_block().var得到所有variable，找到所需变量对应名字。
2. 设置variable.persistable为True，打印variable即可得到名字。

为得到模块的输出，提供如下三种模型调试方法：

1. 通过get\_var和变量名字得到variable，在exe.run中fetch该variable得到对应的值。例如：

        rpn_conv = fluid.get_var('conv_rpn.tmp_1')
        rpn_conv_v = exe.run(fetch_list=rpn_conv)

2. 在exe.run后，通过 global_scope().find_var直接得到变量值。例如：

        rpn_conv_v = fluid.global_scope().find_var('conv_rpn.tmp_1').get_tensor()

3. 直接采用Print op打印得到变量值。例如：

        fluid.layers.Print(rpn_conv)
