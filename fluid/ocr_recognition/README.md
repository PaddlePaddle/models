
[toc]

运行本目录下的程序示例需要使用PaddlePaddle v0.11.0 版本。如果您的PaddlePaddle安装版本低于此要求，请按照安装文档中的说明更新PaddlePaddle安装版本。

# Optical Character Recognition

这里将介绍如何在PaddlePaddle fluid下使用CRNN-CTC 和 CRNN-Attention模型对图片中的文字内容进行识别。

## 1. CRNN-CTC
本章的任务是识别含有单行汉语字符图片，首先采用卷积将图片转为`features map`, 然后使用`im2sequence op`将`features map`转为`sequence`，经过`双向GRU RNN`得到每个step的汉语字符的概率分布。训练过程选用的损失函数为CTC loss，最终的评估指标为`instance error rate`。
本路径下各个文件的作用如下：

- **ctc_reader.py : ** 下载、读取、处理数据。提供方法`train()` 和 `test()` 分别产生训练集和测试集的数据迭代器。
- **crnn_ctc_model.py : ** 在该脚本中定义了训练网络、预测网络和evaluate网络。
- **ctc_train.py : ** 用于模型的训练，可通过命令`python train.py --help` 获得使用方法。
- **inference.py : ** 加载训练好的模型文件，对新数据进行预测。可通过命令`python inference.py --help` 获得使用方法。
- **eval.py : ** 评估模型在指定数据集上的效果。可通过命令`python inference.py --help` 获得使用方法。
-  **utility.py : ** 实现的一些通用方法，包括参数配置、tensor的构造等。


### 1.1 数据
数据的下载和简单预处理都在`ctc_reader.py`中实现。

#### 1.1.1 数据格式

我们使用的训练和测试数据如`图1`所示，每张图片包含单行不定长的中文字符串，这些图片都是经过检测算法进行预框选处理的。

<p align="center">
<img src="images/demo.jpg" width="620" hspace='10'/> <br/>
<strong>图 1</strong>
</p>

在训练集中，每张图片对应的label是由若干数字组成的sequence。 Sequence中的每个数字表示一个字符在字典中的index。 `图1` 对应的label如下所示：
```
3835,8371,7191,2369,6876,4162,1938,168,1517,4590,3793
```
在上边这个label中，`3835` 表示字符‘两’的index，`4590` 表示中文字符逗号的index。


#### 1.1.2 数据准备
**A. 训练集**
我们需要把所有参与训练的图片放入同一个文件夹，暂且记为`train_images`。然后用一个list文件存放每张图片的信息，包括图片大小、图片名称和对应的label，这里暂记该list文件为`train_list`，其格式如下所示：

```
185 48 00508_0215.jpg 7740,5332,2369,3201,4162
48 48 00197_1893.jpg 6569
338 48 00007_0219.jpg 4590,4788,3015,1994,3402,999,4553
150 48 00107_4517.jpg 5936,3382,1437,3382
...
157 48 00387_0622.jpg 2397,1707,5919,1278
```
<center>文件train_list</center>
上述文件中的每一行表示一张图片，每行被空格分为四列，前两列分别表示图片的宽和高，第三列表示图片的名称，第四列表示该图片对应的sequence label。
最终我们应有以下类似文件结构：
```
|-train_data
    |- train_list
    |- train_imags
        |- 00508_0215.jpg
        |- 00197_1893.jpg
        |- 00007_0219.jpg
        | ...
```
在训练时，我们通过选项`--train_images` 和 `--train_list` 分别设置准备好的`train_images` 和`train_list`。

>**注：** 如果`--train_images` 和 `--train_list`都未设置或设置为None， ctc_reader.py会自动下载使用[示例数据](http://cloud.dlnel.org/filepub/?uuid=df937251-3c0b-480d-9a7b-0080dfeee65c)，并将其缓存到`$HOME/.cache/paddle/dataset/ctc_data/data/` 路径下。

**B. 测试集和评估集**
测试集、评估集的准备方式与训练集相同。
在训练阶段，测试集的路径通过train.py的选项`--test_images` 和 `--test_list` 来设置。
在评估时，评估集的路径通过eval.py的选项`--input_images_dir` 和`--input_images_list` 来设置。

**C. 待预测数据集**
待预测数据集的格式与训练集也类似，只不过list文件中的最后一列可以放任意占位字符或字符串，如下所示：

```
185 48 00508_0215.jpg s
48 48 00197_1893.jpg s
338 48 00007_0219.jpg s
...
```
在做inference时，通过inference.py的选项`--input_images_dir`和`--input_images_list` 来设置输入数据的路径。

#### 1.2 训练

通过以下命令调用训练脚本进行训练：
```
python ctc_train.py [options]
```

其中，options支持配置以下训练相关的参数：

  **- -batch_size : ** Minibatch 大小，默认为32.

**- -pass_num : ** 训练多少个pass。默认为100。

  **- -log_period : **  每隔多少个minibatch打印一次训练日志， 默认为1000.

**- -save_model_period : ** 每隔多少个minibatch保存一次模型。默认为15000。
如果设置为-1，则永不保存模型。

   **- -eval_period : ** 每隔多少个minibatch用测试集测试一次模型。默认为15000。如果设置为-1，则永不进行测试。

  **- -save_model_dir :  ** 保存模型文件的路径，默认为“./models”，如果指定路径不存在，则会自动创建路径。

   **- -init_model : ** 初始化模型的路径。如果模型是以单个文件存储的，这里需要指定具体文件的路径；如果模型是以多个文件存储的，这里只需指定多个文件所在文件夹的路径。该选项默认为 None，意思是不用预训练模型做初始化。

   **- -learning_rate : **  全局learning rate. 默认为 0.001.

   **- -l2 :  **  L2 regularizer. 默认为0.0004.

 **- -max_clip  : ** Max gradient clipping threshold. 默认为10.0.

 **- -momentum : ** Momentum. 默认为0.9.

**- -rnn_hidden_size:  ** RNN 隐藏层大小。 默认为200。

**- -device DEVICE : ** 设备ID。设置为-1，训练在CPU执行；设置为0，训练在GPU上执行。默认为0。

 **- -min_average_window : ** Min average window. 默认为10000.

 **- -max_average_window : ** Max average window. 建议大小设置为一个pass内minibatch的数量。默认为15625.

 **- -average_window : ** Average window. 默认为0.15.

  **- -parallel : ** 是否使用多卡进行训练。默认为True.

 **- -train_images : ** 存放训练集图片的路径，如果设置为None，ctc_reader会自动下载使用默认数据集。如果使用自己的数据进行训练，需要修改该选项。默认为None。

**- -train_list : ** 存放训练集图片信息的list文件，如果设置为None，ctc_reader会自动下载使用默认数据集。如果使用自己的数据进行训练，需要修改该选项。默认为None。

  **- -test_images : ** 存放测试集图片的路径，如果设置为None，ctc_reader会自动下载使用默认数据集。如果使用自己的数据进行测试，需要修改该选项。默认为None。

  **- -test_list : ** 存放测试集图片信息的list文件，如果设置为None，ctc_reader会自动下载使用默认数据集。如果使用自己的数据进行测试，需要修改该选项。默认为None。

  **- -num_classes :  ** 字符集的大小。如果设置为None, 则使用ctc_reader提供的字符集大小。如果使用自己的数据进行训练，需要修改该选项。默认为None.


### 1.3 Inference

通过以下命令调用预测脚本进行预测：
```
python inference.py [options]
```

其中，options支持配置以下预测相关的参数：

**--model_path ： **  用来做预测的模型文件。如果模型是以单个文件存储的，这里需要指定具体文件的路径；如果模型是以多个文件存储的，这里只需指定多个文件所在文件夹的路径。为必设置选项。

  **--input_images_dir ： ** 存放待预测图片的文件夹路径。如果设置为None, 则使用ctc_reader提供的默认数据。默认为None.

 **--input_images_list ： **  存放待预测图片信息的list文件的路径。如果设置为None, 则使用ctc_reader提供的默认数据。默认为None.

  **--device DEVICE ：** 设备ID。设置为-1，运行在CPU上；设置为0，运行在GPU上。默认为0。

预测结果会print到标准输出。

### 1.4 Evaluate

通过以下命令调用评估脚本用指定数据集对模型进行评估：
```
python eval.py [options]
```

其中，options支持配置以下评估相关的参数：

**--model_path ： **  待评估模型的文件路径。如果模型是以单个文件存储的，这里需要指定具体文件的路径；如果模型是以多个文件存储的，这里只需指定多个文件所在文件夹的路径。为必设置选项。

  **--input_images_dir ： ** 存放待评估图片的文件夹路径。如果设置为None, 则使用ctc_reader提供的默认数据。默认为None.

 **--input_images_list ： **  存放待评估图片信息的list文件的路径。如果设置为None, 则使用ctc_reader提供的默认数据。默认为None.

  **--device DEVICE ：** 设备ID。设置为-1，运行在CPU上；设置为0，运行在GPU上。默认为0。
