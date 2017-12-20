图像分类
=======================

这里将介绍如何在PaddlePaddle下使用AlexNet、VGG、GoogLeNet、ResNet和Inception-ResNet-v2模型进行图像分类。图像分类问题的描述和这五种模型的介绍可以参考[PaddlePaddle book](https://github.com/PaddlePaddle/book/tree/develop/03.image_classification)。

## 训练模型

### 初始化

在初始化阶段需要导入所用的包，并对PaddlePaddle进行初始化。

```python
import gzip
import argparse

import paddle.v2.dataset.flowers as flowers
import paddle.v2 as paddle
import reader
import vgg
import resnet
import alexnet
import googlenet
import inception_resnet_v2


# PaddlePaddle init
paddle.init(use_gpu=False, trainer_count=1)
```

### 定义参数和输入

设置算法参数（如数据维度、类别数目和batch size等参数），定义数据输入层`image`和类别标签`lbl`。

```python
# Use 3 * 331 * 331 or 3 * 299 * 299 for DATA_DIM in Inception-ResNet-v2.
DATA_DIM = 3 * 224 * 224
CLASS_DIM = 102
BATCH_SIZE = 128

image = paddle.layer.data(
    name="image", type=paddle.data_type.dense_vector(DATA_DIM))
lbl = paddle.layer.data(
    name="label", type=paddle.data_type.integer_value(CLASS_DIM))
```

### 获得所用模型

这里可以选择使用AlexNet、VGG、GoogLeNet、ResNet和Inception-ResNet-v2模型中的一个模型进行图像分类。通过调用相应的方法可以获得网络最后的Softmax层。

1. 使用AlexNet模型

指定输入层`image`和类别数目`CLASS_DIM`后，可以通过下面的代码得到AlexNet的Softmax层。

```python
out = alexnet.alexnet(image, class_dim=CLASS_DIM)
```

2. 使用VGG模型

根据层数的不同，VGG分为VGG13、VGG16和VGG19。使用VGG16模型的代码如下：

```python
out = vgg.vgg16(image, class_dim=CLASS_DIM)
```

类似地，VGG13和VGG19可以分别通过`vgg.vgg13`和`vgg.vgg19`方法获得。

3. 使用GoogLeNet模型

GoogLeNet在训练阶段使用两个辅助的分类器强化梯度信息并进行额外的正则化。因此`googlenet.googlenet`共返回三个Softmax层，如下面的代码所示：

```python
out, out1, out2 = googlenet.googlenet(image, class_dim=CLASS_DIM)
loss1 = paddle.layer.cross_entropy_cost(
    input=out1, label=lbl, coeff=0.3)
paddle.evaluator.classification_error(input=out1, label=lbl)
loss2 = paddle.layer.cross_entropy_cost(
    input=out2, label=lbl, coeff=0.3)
paddle.evaluator.classification_error(input=out2, label=lbl)
extra_layers = [loss1, loss2]
```

对于两个辅助的输出，这里分别对其计算损失函数并评价错误率，然后将损失作为后文SGD的extra_layers。

4. 使用ResNet模型

ResNet模型可以通过下面的代码获取：

```python
out = resnet.resnet_imagenet(image, class_dim=CLASS_DIM)
```

5. 使用Inception-ResNet-v2模型

提供的Inception-ResNet-v2模型支持`3 * 331 * 331`和`3 * 299 * 299`两种大小的输入，同时可以自行设置dropout概率，可以通过如下的代码使用：

```python
out = inception_resnet_v2.inception_resnet_v2(
    image, class_dim=CLASS_DIM, dropout_rate=0.5, size=DATA_DIM)
```

注意，由于和其他几种模型输入大小不同，若配合提供的`reader.py`使用Inception-ResNet-v2时请先将`reader.py`中`paddle.image.simple_transform`中的参数为修改为相应大小。

### 定义损失函数

```python
cost = paddle.layer.classification_cost(input=out, label=lbl)
```

### 创建参数和优化方法

```python
# Create parameters
parameters = paddle.parameters.create(cost)

# Create optimizer
optimizer = paddle.optimizer.Momentum(
    momentum=0.9,
    regularization=paddle.optimizer.L2Regularization(rate=0.0005 *
                                                     BATCH_SIZE),
    learning_rate=0.001 / BATCH_SIZE,
    learning_rate_decay_a=0.1,
    learning_rate_decay_b=128000 * 35,
    learning_rate_schedule="discexp", )
```

通过 `learning_rate_decay_a` (简写$a$） 、`learning_rate_decay_b` (简写$b$) 和 `learning_rate_schedule` 指定学习率调整策略，这里采用离散指数的方式调节学习率，计算公式如下， $n$ 代表已经处理过的累计总样本数，$lr_{0}$ 即为参数里设置的 `learning_rate`。

$$  lr = lr_{0} * a^ {\lfloor \frac{n}{ b}\rfloor} $$


### 定义数据读取

首先以[花卉数据](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)为例说明如何定义输入。下面的代码定义了花卉数据训练集和验证集的输入：

```python
train_reader = paddle.batch(
    paddle.reader.shuffle(
        flowers.train(),
        buf_size=1000),
    batch_size=BATCH_SIZE)
test_reader = paddle.batch(
    flowers.valid(),
    batch_size=BATCH_SIZE)
```

若需要使用其他数据，则需要先建立图像列表文件。`reader.py`定义了这种文件的读取方式，它从图像列表文件中解析出图像路径和类别标签。

图像列表文件是一个文本文件，其中每一行由一个图像路径和类别标签构成，二者以跳格符（Tab）隔开。类别标签用整数表示，其最小值为0。下面给出一个图像列表文件的片段示例：

```
dataset_100/train_images/n03982430_23191.jpeg    1
dataset_100/train_images/n04461696_23653.jpeg    7
dataset_100/train_images/n02441942_3170.jpeg 8
dataset_100/train_images/n03733281_31716.jpeg    2
dataset_100/train_images/n03424325_240.jpeg  0
dataset_100/train_images/n02643566_75.jpeg   8
```

训练时需要分别指定训练集和验证集的图像列表文件。这里假设这两个文件分别为`train.list`和`val.list`，数据读取方式如下：

```python
train_reader = paddle.batch(
    paddle.reader.shuffle(
        reader.train_reader('train.list'),
        buf_size=1000),
    batch_size=BATCH_SIZE)
test_reader = paddle.batch(
    reader.test_reader('val.list'),
    batch_size=BATCH_SIZE)
```

### 定义事件处理程序
```python
# End batch and end pass event handler
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 1 == 0:
            print "\nPass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics)
    if isinstance(event, paddle.event.EndPass):
        with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
            parameters.to_tar(f)

        result = trainer.test(reader=test_reader)
        print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
```

### 定义训练方法

对于AlexNet、VGG、ResNet和Inception-ResNet-v2，可以按下面的代码定义训练方法：

```python
# Create trainer
trainer = paddle.trainer.SGD(
    cost=cost,
    parameters=parameters,
    update_equation=optimizer)
```

GoogLeNet有两个额外的输出层，因此需要指定`extra_layers`，如下所示：

```python
# Create trainer
trainer = paddle.trainer.SGD(
    cost=cost,
    parameters=parameters,
    update_equation=optimizer,
    extra_layers=extra_layers)
```

### 开始训练

```python
trainer.train(
    reader=train_reader, num_passes=200, event_handler=event_handler)
```

## 应用模型
模型训练好后，可以使用下面的代码预测给定图片的类别。

```python
# load parameters
with gzip.open('params_pass_10.tar.gz', 'r') as f:
    parameters = paddle.parameters.Parameters.from_tar(f)

file_list = [line.strip() for line in open(image_list_file)]
test_data = [(paddle.image.load_and_transform(image_file, 256, 224, False)
              .flatten().astype('float32'), )
             for image_file in file_list]
probs = paddle.infer(
    output_layer=out, parameters=parameters, input=test_data)
lab = np.argsort(-probs)
for file_name, result in zip(file_list, lab):
    print "Label of %s is: %d" % (file_name, result[0])
```

首先从文件中加载训练好的模型（代码里以第10轮迭代的结果为例），然后读取`image_list_file`中的图像。`image_list_file`是一个文本文件，每一行为一个图像路径。代码使用`paddle.infer`判断`image_list_file`中每个图像的类别，并进行输出。

## 使用预训练模型
为方便进行测试和fine-tuning，我们提供了一些对应于示例中模型配置的预训练模型，目前包括在ImageNet 1000类上训练的ResNet50、ResNet101和Vgg16，请使用`models`目录下的脚本`model_download.sh`进行模型下载，如下载ResNet50可进入`models`目录并执行"`sh model_download.sh ResNet50`"，完成后同目录下的`Paddle_ResNet50.tar.gz`即是训练好的模型，可以在代码中使用如下两种方式进行加载模：

```python
parameters = paddle.parameters.Parameters.from_tar(gzip.open('Paddle_ResNet50.tar.gz', 'r'))
```

```python
parameters = paddle.parameters.create(cost)
parameters.init_from_tar(gzip.open('Paddle_ResNet50.tar.gz', 'r'))
```

### 注意事项
模型压缩包中所含各文件的文件名和模型配置中的参数名一一对应，是加载模型参数的依据。我们提供的预训练模型均使用了示例代码中的配置，如需修改网络配置，请多加注意，需要保证网络配置中的参数名和压缩包中的文件名能够正确对应。
