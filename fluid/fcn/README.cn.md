**FCN语义分割**
---

**概述**
---
FCN全称：Fully Convolutional Networks for Semantic Segmentation, 是基于深度学习算法完成图像语义分割任务的开创性工作[1]。本示例旨在介绍如何使用PaddlePaddle中的FCN模型进行语义分割。下面首先简要介绍FCN原理，然后介绍示例包含文件及如何使用，接着介绍如何在PASCAL VOC数据集上训练和测试模型。

**FCN原理**
---
FCN基于卷积神经网络实现“端到端”的分割：输入是测试图像，输出为分割结果。论文基于VGG16[2]作为基础网络进行特征提取，不过对基础网络进行了改写以适应图像语义分割任务，具体包含：
1. 将网络中全连接层转化为卷积操作，以接受任意大小的输入图像。
2. 使用转置卷积的方式对特征图进行上采样，以输出和输入图像相同分辨率的特征图。
3. 引入Skip-Connection的连接方式，在网络深层引入浅层信息，以得到更精细的分割结果。

下图为FCN框架：
![FCN框架](https://github.com/chengyuz/models/blob/yucheng/fluid/fcn/images/fcn_network.png?raw=true)

深度网络浅层具有丰富的空间细节信息，而语义信息主要集中于网络深层，由此论文在网络深层引入浅层信息作为补充。具体来说，论文中提出了三个分割模型：FCN-32s，FCN-16s和FCN-8s，FCN-32s直接使用转置卷积的方式对pool5层的输出进行上采样；FCN-16s首先对pool5层的输出进行上采样，然后和pool4层的输出使用sum操作进行特征融合，再进行上采样；FCN-8s引入了更浅层的pool3层信息进行特征融合。

**示例总览**
---
本示例共包含以下文件：

表1. 示例文件

文件                              | 用途                                   |
-------------------------         | -------------------------------------   |
 train.py                          | 训练脚本                                |  
 infer.py                          | 测试脚本，给定图片及模型，完成测试      |  
 vgg_fcn.py                        | FCN模型框架定义脚本                     |  
 data_provider.py                  | 数据处理脚本，生成训练和测试数据        |  
 utils.py                          | 常用函数脚本                            |  
 data/prepare_voc_data.py          | 准备PASCAL VOC训练和测试文件            |  

**PASCAL VOC数据集**
---
**数据准备**

1. 请首先下载数据集：[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)[3]训练集和[VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)[4]测试集。将下载好的数据解压，目录结构为：`data/VOCdevkit/VOC2012`和`data/VOCdevkit/VOC2007`。
2. 进入`data`目录，运行`python prepare_voc_data.py`，即可生成`voc2012_trainval.txt`和`voc2007_test.txt`。

下面是`voc2012_trainval.txt`前几行输入示例：
```
 VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg voc_processed/2007_000032.png
 VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg voc_processed/2007_000033.png
 VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg voc_processed/2007_000039.png
```
下面是`voc2007_test.txt`前几行输入示例：
```
VOCdevkit/VOC2007/JPEGImages/000068.jpg
VOCdevkit/VOC2007/JPEGImages/000175.jpg
VOCdevkit/VOC2007/JPEGImages/000243.jpg
```

**预训练模型准备**

下载预训练的VGG16模型，我们提供了一个转化好的模型，下载地址：[VGG16](https://pan.baidu.com/s/1ekZ5O-lp3lGvAOZ4KSXKDQ)，将其放置到：`models/vgg16_weights.tar`, 然后解压用于初始化。

**模型训练**

直接执行`python train.py --fcn_arch fcn-32s`即可训练FCN-32s模型，现在同时支持FCN-16s和FCN-8s分割模型。`train.py`中关键逻辑：
```python
weights_dict = resolve_caffe_model(args.pretrain_model)
for k, v in weights_dict.items():
    _tensor = fluid.global_scope().find_var(k).get_tensor()
    _shape = np.array(_tensor).shape
    _tensor.set(v, place)

data_args = data_provider.Settings(
        data_dir=args.data_dir,
        resize_h=args.img_height,
        resize_w=args.img_width,
        mean_value=mean_value)
```
主要包括：
1. 调用`resolve_caffe_model`得到预训练模型参数，然后基于fluid中tensor的`set`函数为模型赋初值。
2. 调用`data_provider.Settings`配置数据预处理参数，运行时可通过命令行对相应参数进行配置。
3. 训练中每隔一定epoch会调用`fluid.io.save_inference_model`存储模型。

下面给出了FCN-32s，FCN-16s和FCN-8s在VOC数据集上训练的Loss曲线：

![FCN训练损失曲线](https://github.com/chengyuz/models/blob/yucheng/fluid/fcn/images/train_loss.png?raw=true)

**模型测试**

执行`python infer.py --fcn_arch fcn-32s`即可使用训练好的FCN-32s模型对输入图片进行分割，预测结果保存在`demo`文件夹，具体可通过`--vis_dir`进行配置。`infer.py`中关键逻辑：
```python
model_dir = os.path.join(args.model_dir, '%s-model' % args.fcn_arch)
assert(os.path.exists(model_dir))
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_dir, exe)

predict = exe.run(inference_program, feed={feed_target_names[0]:img_data}, fetch_list=fetch_targets)
res = np.argmax(np.squeeze(predict[0]), axis=0)
res = convert_to_color_label(res)
```
主要包括：
1. 调用`fluid.io.load_inference_model`加载训练好的模型。
2. 调用`convert_to_color_label`将模型预测结果可视化为VOC对应格式。

下图是FCN-32s，FCN-16s和FCN-8s的部分测试结果：

![FCN-32s分割结果](https://github.com/chengyuz/models/blob/yucheng/fluid/fcn/images/seg_res.png?raw=true)

我们提供了训练好的模型用于测试：
[FCN-32s](https://pan.baidu.com/s/1j8pltdzgssmxbXFgHWmCNQ)（密码：dk0i）
[FCN-16s](https://pan.baidu.com/s/1idapCRSxWsJKSqqswUGDSw)（密码：q8gu）
[FCN-8s](https://pan.baidu.com/s/1GcO-mcOWo_VF65X3xwPnpA)（密码：du9x）

**引用**
---
1. Jonathan Long, Evan Shelhamer, Trevor Darrell. [Fully convolutional networks for semantic segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). IEEE conference on computer vision and pattern recognition, 2015.
2. Simonyan, Karen, and Andrew Zisserman. [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/abs/1409.1556). arXiv preprint arXiv:1409.1556 (2014).
3. [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
4. [The PASCAL Visual Object Classes Challenge 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)
