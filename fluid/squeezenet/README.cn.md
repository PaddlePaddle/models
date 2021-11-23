SqueezeNet
===========
Introduction
-----------
近来深层卷积网络的主要研究方向集中在提高正确率。对于相同的正确率水平，更小的CNN架构可以提供如下的优势：
* 在分布式训练中，与服务器通信需求更小
* 参数更少，从云端下载模型的数据量小
* 更适合在FPGA等内存受限的设备上部署。
基于这些优点, Squeezenet 提出fire module 模块，它在ImageNet上实现了和AlexNet相同的正确率，但是只使用了1/50的参数。
更进一步，使用模型压缩技术，可以将SqueezeNet压缩到0.5MB，这是AlexNet的1/510。

Architecture
-----------
### Architecture Design Strategies
  * 使用1∗1卷积代替3∗3 卷积：参数减少为原来的1/9
  * 减少输入通道数量：这一部分使用squeeze layers来实现
  * 将欠采样操作延后，可以给卷积层提供更大的激活图：更大的激活图保留了更多的信息，可以提供更高的分类准确率
### The Fire Module
![](https://github.com/Panxj/SqueezeNet/raw/master/images/fire_module.jpg)
  * squeeze layer: 使用`1∗1`卷积核压缩通道数
  * expand layer: 分别使用 `1∗1` 与 `3∗3` 卷积核对扩展通道数
  * Fire module中使用3个可调的超参数：`s1x1`（squeeze convolution layer中1∗1 filter的个数）、`e1x1`（expand layer中1∗1 filter的个数）、`e3x3`（expand layer中3∗3 filter的个数）
  * 使用Fire module的过程中，令`s1x1 < e1x1 + e3x3`，这样squeeze layer可以限制输入通道数量

### The SqueezeNet Architecture
SqueezeNet以卷积层（conv1）开始，接着使用8个Fire modules (fire2-9)，最后以卷积层（conv10）结束。每个fire module中的filter数量逐渐增加，并且在conv1, fire4, fire8, 和 conv10这几层之后使用步长为2的max-pooling，即将池化层放在相对靠后的位置，如下图左侧子图，中间与右侧子图分别在初始结构上添加
simple bypass 与 complex bypass.

![](https://github.com/Panxj/SqueezeNet/raw/master/images/architecture.jpg)

Overview
-----------
Tabel 1. Directory structure

|file | description|
|:--- |:---|
train.py | Train script
infer.py | Prediction using the trained model
reader.py| Data reader
squeezenet.py| Model definition
data/val.txt|Validation data list
data/train.txt| Train data list

Data Preparation
-----------
首先从官网下载imagenet数据集，使用ILSVRC 2012(ImageNet Large Scale Visual Recognition Challenge)比赛用的子数据集，其中<br>
* 训练集: 1,281,167张图片 + 标签
* 验证集: 50,000张图片 + 标签
* 测试集: 100,000张图片

训练时， 所有图片resize到 `256 X 256`，之后随机crop 出 `227 X 227` 大小图像输入网络。验证与测试时，同样首先resize到 `256 X 256`，之后从中间 crop 出 `227 X 227` 图像输入网络。所有图像均减去均值`[104,117,123]`，与imagenet 官网提供的均值文件稍有不同。
`reader.py`中相关函数如下，
```python
def process_image(sample, mode, color_jitter, rotate):
    img_path = sample[0]
    img = paddle2.image.load_image(img_path)
    img = cv2.resize(img, (DATA_DIM, DATA_DIM), interpolation=cv2.INTER_CUBIC)
    if mode == 'train':
        if rotate: img = rotate_image(img)
        img = paddle2.image.random_crop(img, DATA_DIM)
    else:
        img = paddle2.image.center_crop(img, DATA_DIM)
    if mode == 'train':
        if color_jitter:
            img = distort_color(img)
        if random.randint(0, 1) == 1:
            img = paddle2.image.left_right_flip(img)
    img = paddle2.image.to_chw(img)
    img = img.astype('float32')
    img -= img_mean

    if mode == 'train' or mode == 'test':
        return img, sample[1]
    elif mode == 'infer':
        return [img]
```

train.txt 中数据如下：
```
n01440764/n01440764_10026.JPEG 0
n01440764/n01440764_10027.JPEG 0
n01440764/n01440764_10029.JPEG 0
n01440764/n01440764_10040.JPEG 0
n01440764/n01440764_10042.JPEG 0
n01440764/n01440764_10043.JPEG 0
```
val.txt 中数据如下：
```
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
ILSVRC2012_val_00000003.JPEG 230
ILSVRC2012_val_00000004.JPEG 809
ILSVRC2012_val_00000005.JPEG 516
ILSVRC2012_val_00000006.JPEG 57
```
synset_wrods.txy 数据如下：
```
n01491361 tiger shark, Galeocerdo cuvieri
n01494475 hammerhead, hammerhead shark
n01496331 electric ray, crampfish, numbfish, torpedo
```

Training
-----------
#### 1. Determine the architecture
论文作者[github](https://github.com/DeepScale/SqueezeNet)开源了两个版本的SqueezeNet 模型。 其中 SqueezeNet_v1.0 与论文中结构相同，SqueezeNet_v1.1 对原有结构进行了些许改动，使得在保证accuracy 不下降的情况下，计算量降低了 2.4x 倍。 SqueezeNet_v1.1 相比于论文中结构改动如下：

Tabel 2. changes in SqueezeNet_v1.1

 | | SqueezeNet_v1.0 | SqueezeNet_v1.1|
 |:---|:---|:---|
 |conv1| 96 filters of resolution 7x7|64 filters of resolution 3x3|
 |pooling layers| pool_{1,4,8} | pool_{1,3,5}|
 |computation| 1.72GFLOPS/image| 0.72 GFOLPS/image:2.4x less computation|
 |ImageNet accuracy| >=80.3% top-5| >=80.3% top-5|

此项目中，采用SqueezeNet_v1.1 结构。<br>

<!--#### caffe2paddle 参数转化
caffemodel中参数按照[PaddlePaddle](https://github.com/PaddlePaddle/models/tree/develop/image_classification/caffe2paddle)介绍的方法进行转化，并在训练开始前进行赋值，如下：
```python
#Load pre-trained params
if args.model is not None:
    for layer_name in parameters.keys():
        layer_param_path = os.path.join(args.model,layer_name)
        if os.path.exists(layer_param_path):
            h,w = parameters.get_shape(layer_name)
            parameters.set(layer_name,load_parameter(layer_param_path,h,w))
```
-->
#### 2. train
`python train.py | tee ouput/logs/log.log` 执行训练过程。


```python
train_parallel_do(args,
                      learning_rate,
                      batch_size,
                      num_passes,
                      init_model=None,
                      pretrained_model=None,
                      model_save_dir='models',
                      parallel=True,
                      use_nccl=True,
                      lr_strategy=None)
```

Testing
-----------
Run `python eavl.py` 执行测试过程.
```python
add_arg('batch_size', int, 32, "Minibatch size.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('test_list', str, '', "The testing data lists.")
add_arg('model_dir', str, './models/final', "The model path.")

# Evaluation code
eval(args):
```

 `args.test_list` 指定测试过程中的图片路径列表,  `args.model_path` 指定已训练好的模型路径。
```
测试结果如下.
```

Infering
-----------
Run `python infer.py` 利用训练好的模型进行推断.
```python
add_arg('batch_size', int, 1, "Minibatch size.")
add_arg('use_gpu', bool,  True, "Whether to use GPU or not.")
add_arg('test_list', str, '', "The testing data lists.")
add_arg('synset_word_list', str, 'data/ILSVRC2012/synset_words.txt', "The label name of data")
add_arg('model_dir', str, 'models/final', "The model path.")
# infer code
infer(args)
```

`args.test_list` 指定需要参与推断的图片路径列表文件, `args.model_path` 指定用到的训练好的模型.
```
infer example result.
```

References
-----------
[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
