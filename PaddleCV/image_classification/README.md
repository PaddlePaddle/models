中文 | [English](README_en.md)

# 图像分类以及模型库


> 注意：该图像分类库已迁移至新的github地址：[https://github.com/PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas)，欢迎大家去新的代码仓库中，查看与阅读更多关于图像分类的详细介绍以及新功能。

## 内容
- [简介](#简介)
- [快速开始](#快速开始)
    - [安装说明](#安装说明)
    - [数据准备](#数据准备)
    - [模型训练](#模型训练)
    - [参数微调](#参数微调)
    - [模型评估](#模型评估)
    - [模型预测](#模型预测)
- [进阶使用](#进阶使用)
    - [Mixup训练](#mixup训练)
    - [混合精度训练](#混合精度训练)
    - [性能分析](#性能分析)
    - [DALI预处理](#DALI预处理)
- [已发布模型及其性能](#已发布模型及其性能)
- [FAQ](#faq)
- [参考文献](#参考文献)
- [版本更新](#版本更新)
- [如何贡献代码](#如何贡献代码)

---

## 简介
图像分类是计算机视觉的重要领域，它的目标是将图像分类到预定义的标签。近期，许多研究者提出很多不同种类的神经网络，并且极大的提升了分类算法的性能。本页将介绍如何使用PaddlePaddle进行图像分类。

同时推荐用户参考[ IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/122278)

## 快速开始

### 安装说明

在当前目录下运行样例代码需要python 2.7及以上版本，PadddlePaddle Fluid v1.6或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据 [安装文档](https://www.paddlepaddle.org.cn/install/quick) 中的说明来更新PaddlePaddle。

#### 环境依赖

python >= 2.7
运行训练代码需要安装numpy，cv2

```bash
pip install opencv-python
pip install numpy
```

### 数据准备

下面给出了ImageNet分类任务的样例，
在Linux系统下通过如下的方式进行数据的准备：
```
cd data/ILSVRC2012/
sh download_imagenet2012.sh
```
在```download_imagenet2012.sh```脚本中，通过下面三步来准备数据：

**步骤一：** 首先在```image-net.org```网站上完成注册，用于获得一对```Username```和```AccessKey```。

**步骤二：** 从ImageNet官网下载ImageNet-2012的图像数据。训练以及验证数据集会分别被下载到"train" 和 "val" 目录中。注意，ImageNet数据的大小超过140GB，下载非常耗时；已经自行下载ImageNet的用户可以直接将数据组织放置到```data/ILSVRC2012```。

**步骤三：** 下载训练与验证集合对应的标签文件。下面两个文件分别包含了训练集合与验证集合中图像的标签：

* train_list.txt: ImageNet-2012训练集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
train/n02483708/n02483708_2436.jpeg 369
```
* val_list.txt: ImageNet-2012验证集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
val/ILSVRC2012_val_00000001.jpeg 65
```
注意：可能需要根据本地环境调整reader.py中相关路径来正确读取数据。

**Windows系统下请用户自行下载ImageNet数据，[label下载链接](http://paddle-imagenet-models.bj.bcebos.com/ImageNet_label.tgz)**

### 模型训练

数据准备完毕后，可以通过如下的方式启动训练：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fraction_of_gpu_memory_to_use=0.98

python train.py \
        --data_dir=./data/ILSVRC2012/ \
        --total_images=1281167 \
        --class_dim=1000 \
        --validate=True \
        --model=ResNet50_vd \
        --batch_size=256 \
        --lr_strategy=cosine_decay \
        --lr=0.1 \
        --num_epochs=200 \
        --model_save_dir=output/ \
        --l2_decay=7e-5 \
        --use_mixup=True \
        --use_label_smoothing=True \
        --label_smoothing_epsilon=0.1
```


注意:
- 当添加如step_epochs这种列表型参数，需要去掉"="，如：--step_epochs 10 20 30
- 如果需要训练自己的数据集，则需要修改根据自己的数据集修改`data_dir`, `total_images`, `class_dim`参数；如果因为GPU显存不够而需要调整`batch_size`，则参数`lr`也需要根据`batch_size`进行线性调整。
- 如果需要使用其他模型进行训练，则需要修改`model`参数，也可以在`scripts/train/`文件夹中根据对应模型的默认运行脚本进行修改并训练。


或通过run.sh 启动训练

```bash
bash run.sh train 模型名
```

**多进程模型训练：**

如果你有多张GPU卡的话，我们强烈建议你使用多进程模式来训练模型，这会极大的提升训练速度。启动方式如下：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch train.py \
       --model=ResNet50 \
       --batch_size=256 \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape 3 224 224 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --reader_thread=4 \
       --lr=0.1
```
或者参考 scripts/train/ResNet50_dist.sh

**参数说明：**

环境配置部分：

* **data_dir**: 数据存储路径，默认值: "./data/ILSVRC2012/"
* **model_save_dir**: 模型存储路径，默认值: "output/"
* **pretrained_model**: 加载预训练模型路径，默认值: None
* **checkpoint**: 加载用于继续训练的检查点（指定具体模型存储路径，如"output/ResNet50/100/"），默认值: None
* **print_step**: 打印训练信息的batch步数，默认值：10
* **save_step**: 保存模型的epoch步数，默认值：1

模型类型和超参配置：

* **model**: 模型名称， 默认值: "ResNet50"
* **total_images**: 图片数，ImageNet2012，默认值: 1281167
* **class_dim**: 类别数，默认值: 1000
* **image_shape**: 图片大小，默认值: [3,224,224]
* **num_epochs**: 训练回合数，默认值: 120
* **batch_size**: batch size大小(所有设备)，默认值: 8
* **test_batch_size**: 测试batch大小，默认值：16
* **lr_strategy**: 学习率变化策略，默认值: "piecewise_decay"
* **lr**: 初始学习率，默认值: 0.1
* **l2_decay**: l2_decay值，默认值: 1e-4
* **momentum_rate**: momentum_rate值，默认值: 0.9
* **step_epochs**: piecewise dacay的decay step，默认值：[30,60,90]
* **decay_epochs**: exponential decay的间隔epoch数, 默认值: 2.4.
* **decay_rate**: exponential decay的下降率, 默认值: 0.97.

数据读取器和预处理配置：

* **lower_scale**: 数据随机裁剪处理时的lower scale值， upper scale值固定为1.0，默认值:0.08
* **lower_ratio**: 数据随机裁剪处理时的lower ratio值，默认值:3./4.
* **upper_ratio**: 数据随机裁剪处理时的upper ratio值，默认值:4./3.
* **resize_short_size**: 指定数据处理时改变图像大小的短边值，默认值: 256
* **use_mixup**: 是否对数据进行mixup处理，默认值: False
* **mixup_alpha**: 指定mixup处理时的alpha值，默认值: 0.2
* **use_aa**: 是否对数据进行auto augment处理. 默认值: False.
* **reader_thread**: 多线程reader的线程数量，默认值: 8
* **reader_buf_size**: 多线程reader的buf_size， 默认值: 2048
* **interpolation**: 插值方法， 默认值：None
* **image_mean**: 图片均值，默认值：[0.485, 0.456, 0.406]
* **image_std**: 图片std，默认值：[0.229, 0.224, 0.225]


一些开关：

* **validate**: 是否在模型训练过程中启动模型测试，默认值: True
* **use_gpu**: 是否在GPU上运行，默认值: True
* **use_label_smoothing**: 是否对数据进行label smoothing处理，默认值: False
* **label_smoothing_epsilon**: label_smoothing的epsilon， 默认值:0.1
* **padding_type**: efficientNet中卷积操作的padding方式, 默认值: "SAME".
* **use_se**: efficientNet中是否使用Squeeze-and-Excitation模块, 默认值: True.
* **use_ema**: 是否在更新模型参数时使用ExponentialMovingAverage. 默认值: False.
* **ema_decay**: ExponentialMovingAverage的decay rate. 默认值: 0.9999.


性能分析：

* **enable_ce**: 是否开启CE，默认值: False
* **random_seed**: 随机数种子，当设置数值后，所有随机化会被固定，默认值: None
* **is_profiler**: 是否开启性能分析，默认值: 0
* **profilier_path**: 分析文件保存位置，默认值: 'profiler_path/'
* **max_iter**: 最大训练batch数，默认值: 0
* **same_feed**: 是否feed相同数据进入网络，设定具体数值来指定数据数量，默认值：0


**数据读取器说明：** 数据读取器定义在```reader.py```文件中，现在默认基于cv2的数据读取器， 在[训练阶段](#模型训练)，默认采用的增广方式是随机裁剪与水平翻转， 而在[模型评估](#模型评估)与[模型预测](#模型预测)阶段用的默认方式是中心裁剪。当前支持的数据增广方式有：

* 旋转
* 颜色抖动
* 随机裁剪
* 中心裁剪
* 长宽调整
* 水平翻转
* 自动增广


### 参数微调

参数微调(Finetune)是指在特定任务上微调已训练模型的参数。可以下载[已发布模型及其性能](#已发布模型及其性能)并且设置```path_to_pretrain_model```为模型所在路径，微调一个模型可以采用如下的命令：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fraction_of_gpu_memory_to_use=0.98

python train.py \
        --data_dir=./data/ILSVRC2012/ \
        --total_images=1281167 \
        --class_dim=1000 \
        --validate=True \
        --model=ResNet50_vd \
        --batch_size=256 \  
        --lr=0.1 \
        --num_epochs=200 \
        --model_save_dir=output/ \
        --l2_decay=7e-5 \
        --pretrained_model=${path_to_pretrain_model} \
        --finetune_exclude_pretrained_params=fc_0.w_0,fc_0.b_0
```

注意：
- 在自己的数据集上进行微调时，则需要修改根据自己的数据集修改`data_dir`, `total_images`, `class_dim`参数。
- 加载的参数是ImageNet1000的预训练模型参数，对于相同模型，最后的类别数或者含义可能不同，因此在加载预训练模型参数时，需要过滤掉最后的FC层，否则可能会因为**维度不匹配**而报错。


### 模型评估

模型评估(Eval)是指对训练完毕的模型评估各类性能指标。可以下载[已发布模型及其性能](#已发布模型及其性能)并且设置```path_to_pretrain_model```为模型所在路径，```json_path```为保存指标的路径。运行如下的命令，可以获得模型top-1/top-5精度。

**参数说明**

* **save_json_path**: 是否将eval结果保存到json文件中，默认值：None
* `model`: 模型名称，与预训练模型需保持一致。
* `batch_size`: 每个minibatch评测的图片个数。
* `data_dir`: 数据路径。注意：该路径下需要同时包括待评估的**图片文件**以及图片和对应类别标注的**映射文本文件**，文本文件名称需为`val.txt`。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fraction_of_gpu_memory_to_use=0.98

python eval.py \
       --model=ResNet50_vd \
       --pretrained_model=${path_to_pretrain_model} \
       --data_dir=./data/ILSVRC2012/ \
       --save_json_path=${json_path} \
       --batch_size=256
```

### 指数滑动平均的模型评估

注意: 如果你使用指数滑动平均来训练模型(--use_ema=True)，并且想要评估指数滑动平均后的模型，需要使用ema_clean.py将训练中保存下来的ema模型名字转换成原始模型参数的名字。

```
python ema_clean.py \
       --ema_model_dir=your_ema_model_dir \
       --cleaned_model_dir=your_cleaned_model_dir

python eval.py \
       --model=ResNet50_vd \
       --pretrained_model=your_cleaned_model_dir
```

### 模型fluid预测

模型预测(Infer)可以获取一个模型的预测分数或者图像的特征，可以下载[已发布模型及其性能](#已发布模型及其性能)并且设置```path_to_pretrain_model```为模型所在路径，```test_res_json_path```为模型预测结果保存的文本路径，```image_path```为模型预测的图片路径或者图片列表所在的文件夹路径。

**参数说明：**

* **data_dir**: 数据存储位置，默认值：`/data/ILSVRC2012/val/`
* **save_inference**: 是否保存二进制模型，默认值：`False`
* **topk**: 按照置信由高到低排序标签结果，返回的结果数量，默认值：1
* **class_map_path**: 可读标签文件路径，默认值：`./utils/tools/readable_label.txt`
* **image_path**: 指定单文件进行预测，默认值：`None`
* **save_json_path**: 将预测结果保存到json文件中，默认值: `test_res.json`

#### 单张图片预测

```bash
export CUDA_VISIBLE_DEVICES=0

python infer.py \
        --model=ResNet50_vd \
        --class_dim=1000  \
        --pretrained_model=${path_to_pretrain_model} \
        --class_map_path=./utils/tools/readable_label.txt \
        --image_path=${image_path} \
        --save_json_path=${test_res_json_path}
```

#### 图片列表预测
* 该种情况下，需要指定```data_dir```路径和```batch_size```。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python infer.py \
        --model=ResNet50_vd \
        --class_dim=1000  \
        --pretrained_model=${path_to_pretrain_model} \
        --class_map_path=./utils/tools/readable_label.txt \
        --data_dir=${data_dir} \
        --save_json_path=${test_res_json_path} \
        --batch_size=${batch_size}
```

注意：
- 模型名称需要与该模型训练时保持一致。
- 模型预测默认ImageNet1000类类别，预测数值和可读标签的map文件存储在`./utils/tools/readable_label.txt`中，如果使用自定义数据，请指定`--class_map_path`参数。


### Python预测API
* Fluid提供了高度优化的C++预测库，为了方便使用，Paddle也提供了C++预测库对应的Python接口，更加具体的Python预测API介绍可以参考这里：[https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/python_infer_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/python_infer_cn.html)
* 使用Python预测API进行模型预测的步骤有模型转换和模型预测，详细介绍如下。

#### 模型转换
* 首先将保存的fluid模型转换为二进制模型，转换方法如下，其中```path_to_pretrain_model```表示预训练模型的路径。

```bash
python infer.py \
        --model=ResNet50_vd \
        --pretrained_model=${path_to_pretrain_model} \
        --save_inference=True
```

注意：
- 预训练模型和模型名称需要保持一致。
- 在转换模型时，使用`save_inference_model`函数进行模型转换，参数`feeded_var_names`表示模型预测时所需提供数据的所有变量名称；参数`target_vars`表示模型的所有输出变量，通过这些输出变量即可得到模型的预测结果。
- 转换完成后，会在`ResNet50_vd`文件下生成`model`和`params`文件。

#### 模型预测

根据转换的模型二进制文件，基于Python API的预测方法如下，其中```model_path```表示model文件的路径，```params_path```表示params文件的路径，```image_path```表示图片文件的路径。

```bash
python predict.py \
        --model_file=./ResNet50_vd/model \
        --params_file=./ResNet50_vd/params \
        --image_path=${image_path} \
        --gpu_id=0 \
        --gpu_mem=1024
```

注意：
- 这里只提供了预测单张图片的脚本，如果需要预测文件夹内的多张图片，需要自己修改预测文件`predict.py`。
- 参数`gpu_id`指定了当前使用的GPU ID号，参数`gpu_mem`指定了初始化的GPU显存。

## 进阶使用

### Mixup训练

训练中指定 --use_mixup=True 开启Mixup训练，本模型库中所有后缀为_vd的模型即代表开启Mixup训练

Mixup相关介绍参考[mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

### 混合精度训练

通过指定--use_fp16=True 启动混合精度训练，在训练过程中会使用float16数据类型，并输出float32的模型参数。您可能需要同时传入--scale_loss来解决fp16训练的精度问题，如传入--scale_loss=128.0。

在配置好数据集路径后（修改[scripts/train/ResNet50_fp16.sh](scripts/train/ResNet50_fp16.sh)文件中`DATA_DIR`的值），对ResNet50模型进行混合精度训练可通过运行`bash run.sh train ResNet50_fp16`命令完成。

多机多卡ResNet50模型的混合精度训练请参考[PaddlePaddle/Fleet](https://github.com/PaddlePaddle/Fleet/tree/develop/benchmark/collective/resnet)。

使用Tesla V100单机8卡、2机器16卡、4机器32卡，对ResNet50模型进行混合精度训练的结果如下（开启DALI）：

* BatchSize = 256

节点数*卡数|吞吐|加速比|test\_acc1|test\_acc5
---|---|---|---|---
1*1|1035 ins/s|1|0.75333|0.92702
1*8|7840 ins/s|7.57|0.75603|0.92771
2*8|14277 ins/s|13.79|0.75872|0.92793
4*8|28594 ins/s|27.63|0.75253|0.92713

* BatchSize = 128

节点数*卡数|吞吐|加速比|test\_acc1|test\_acc5
---|---|---|---|---
1*1|936  ins/s|1|0.75280|0.92531
1*8|7108 ins/s|7.59|0.75832|0.92771
2*8|12343 ins/s|13.18|0.75766|0.92723
4*8|24407 ins/s|26.07|0.75859|0.92871

### 性能分析

注意：本部分主要为内部测试功能。
其中包括启动CE以监测模型运行的稳定性，启动profiler以测试benchmark，启动same_feed来进行快速调试。

启动CE会固定随机初始化，其中包括数据读取器中的shuffle和program的[random_seed](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/fluid_cn/Program_cn.html#random_seed)

``` bash
python train.py \
    --enable_ce=True \
    --data_dir=${path_to_a_smaller_dataset}
```

启动profiler进行性能分析

``` bash
python train.py \
    --is_profiler=True
```

设置same_feed参数以进行快速调试, 相同的图片（same_feed张图片）将传入网络中
```bash
python train.py \
    --same_feed=8 \
    --batch_size=4 \
    --print_step=1
```

### DALI预处理

使用[Nvidia DALI](https://github.com/NVIDIA/DALI)预处理类库可以加速训练并提高GPU利用率。

DALI预处理目前支持标准ImageNet处理步骤（ random crop -> resize -> flip -> normalize），并且支持列表文件或者文件夹方式的数据集格式。

指定`--use_dali=True`即可开启DALI预处理，如下面的例子中，使用DALI训练ShuffleNet v2 0.25x，在8卡v100上，图片吞吐可以达到10000张/秒以上，GPU利用率在85%以上。

``` bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_fraction_of_gpu_memory_to_use=0.80

python -m paddle.distributed.launch train.py \
       --model=ShuffleNetV2_x0_25 \
       --batch_size=2048 \
       --lr_strategy=cosine_decay_warmup \
       --num_epochs=240 \
       --lr=0.5 \
       --l2_decay=3e-5 \
       --lower_scale=0.64 \
       --lower_ratio=0.8 \
       --upper_ratio=1.2 \
       --use_dali=True
```

更多DALI相关用例请参考[DALI Paddle插件文档](https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/plugins/paddle_tutorials.html)。

#### 注意事项

1. 请务必使用GCC5.4以上编译器[编译安装](https://www.paddlepaddle.org.cn/install/doc/source/ubuntu)的1.6或以上版本paddlepaddle, 另外，请在编译过程中指定-DWITH_DISTRIBUTE=ON 来启动多进程训练模式。注意：官方的paddlepaddle是GCC4.8编译的，请务必检查此项，或参考使用[已经编译好的whl包](https://github.com/NVIDIA/DALI/blob/master/qa/setup_packages.py#L38)
2. Nvidia DALI需要使用[#1371](https://github.com/NVIDIA/DALI/pull/1371)以后的git版本。请参考[此文档](https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/installation.html)安装nightly版本或从源码安装。
3. 因为DALI使用GPU进行图片预处理，需要占用部分显存，请适当调整 `FLAGS_fraction_of_gpu_memory_to_use`环境变量（如`0.8`）来预留部分显存供DALI使用。


## 已发布模型及其性能
表格中列出了在models目录下目前支持的图像分类模型，并且给出了已完成训练的模型在ImageNet-2012验证集合上的top-1和top-5精度，以及Paddle Fluid和Paddle TensorRT基于动态链接库的预测时间（测试GPU型号为NVIDIA® Tesla® P4）。
可以通过点击相应模型的名称下载对应的预训练模型。

#### 注意事项

- 特殊参数配置

<table>
<tr>
    <td><b>Model</b>
    </td>
    <td><b>输入图像分辨率</b>
    </td>
    <td><b>参数 resize_short_size</b>
    </td>
</tr>
<tr>
   <td>Inception, Xception
   </td>
   <td>299
   </td>
   <td>320
   </td>
</tr>
<tr>
    <td> DarkNet53
    </td>
    <td>256
    </td>
    <td>256
    </td>
</tr>
<tr>
   <td>Fix_ResNeXt101_32x48d_wsl
   </td>
   <td>320
   </td>
   <td>320
   </td>
</tr>
<tr>
   <td rowspan="8"> EfficientNet: <br/><br/>
   预测时的resize_short_size在其分辨率的长或高的基础上加32<br/>
   在该系列模型训练和预测的过程中<br/>
   图片resize参数interpolation的值设置为2（cubic插值方式）<br/>
   该模型在训练过程中使用了指数滑动平均策略<br/>
   具体请参考<a href="https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/optimizer_cn.html#exponentialmovingaverage">指数滑动平均</a>
   </td>
   <td>B0: 224
   </td>
   <td>256
   </td>
</tr>
<tr>
   <td>B1: 240
   </td>
   <td>272
   </td>
</tr>
<tr>
   <td>B2: 260
   </td>
   <td>292
   </td>
</tr>
<tr>
   <td>B3: 300
   </td>
   <td>332
   </td>
</tr>
<tr>
   <td>B4: 380
   </td>
   <td>412
   </td>
</tr>
<tr>
   <td>B5: 456
   </td>
   <td>488
   </td>
</tr>
<tr>
   <td>B6: 528
   </td>
   <td>560
   </td>
</tr>
<tr>

   <td>B7: 600
   </td>
   <td>632
   </td>
</tr>
<tr>
    <td>其余分类模型
    </td>
    <td>224
    </td>
    <td>256
    </td>
</tr>
</table>


- 调用动态链接库预测时需要将训练模型转换为二进制模型。

    ```bash
    python infer.py \
           --model=ResNet50_vd \
           --pretrained_model=${path_to_pretrain_model} \
           --save_inference=True
    ```

- ResNeXt101_wsl系列的预训练模型转自pytorch模型，详情见[ResNeXt wsl](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/)。


### AlexNet
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[AlexNet](http://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.tar) | 56.72% | 79.17% | 3.083 | 2.566 |

### SqueezeNet
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[SqueezeNet1_0](https://paddle-imagenet-models-name.bj.bcebos.com/SqueezeNet1_0_pretrained.tar) | 59.60% | 81.66% | 2.740 | 1.719 |
|[SqueezeNet1_1](https://paddle-imagenet-models-name.bj.bcebos.com/SqueezeNet1_1_pretrained.tar) | 60.08% | 81.85% | 2.751 | 1.282 |

### VGG Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[VGG11](https://paddle-imagenet-models-name.bj.bcebos.com/VGG11_pretrained.tar) | 69.28% | 89.09% | 8.223 | 6.619 |
|[VGG13](https://paddle-imagenet-models-name.bj.bcebos.com/VGG13_pretrained.tar) | 70.02% | 89.42% | 9.512 | 7.566 |
|[VGG16](https://paddle-imagenet-models-name.bj.bcebos.com/VGG16_pretrained.tar) | 72.00% | 90.69% | 11.315 | 8.985 |
|[VGG19](https://paddle-imagenet-models-name.bj.bcebos.com/VGG19_pretrained.tar) | 72.56% | 90.93% | 13.096 | 9.997 |

### MobileNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[MobileNetV1_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_25_pretrained.tar) | 51.43% | 75.46% | 2.283 | 0.838 |
|[MobileNetV1_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_5_pretrained.tar) | 63.52% | 84.73% | 2.378 | 1.052 |
|[MobileNetV1_x0_75](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_75_pretrained.tar) | 68.81% | 88.23% | 2.540 | 1.376 |
|[MobileNetV1](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) | 70.99% | 89.68% | 2.609 |1.615 |
|[MobileNetV2_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_25_pretrained.tar) | 53.21% | 76.52% | 4.267 | 2.791 |
|[MobileNetV2_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_5_pretrained.tar) | 65.03% | 85.72% | 4.514 | 3.008 |
|[MobileNetV2_x0_75](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_75_pretrained.tar) | 69.83% | 89.01% | 4.313 | 3.504 |
|[MobileNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) | 72.15% | 90.65% | 4.546 | 3.874 |
|[MobileNetV2_x1_5](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x1_5_pretrained.tar) | 74.12% | 91.67% | 5.235 | 4.771 |
|[MobileNetV2_x2_0](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x2_0_pretrained.tar) | 75.23% | 92.58% | 6.680 | 5.649 |
|[MobileNetV3_small_x1_0](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_pretrained.tar) | 67.46% | 87.12% | 6.809 |  |

### ShuffleNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[ShuffleNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar) | 68.80% | 88.45% | 6.101 | 3.616 |
|[ShuffleNetV2_x0_25](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_25_pretrained.tar) | 49.90% | 73.79% | 5.956 | 2.505 |
|[ShuffleNetV2_x0_33](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_33_pretrained.tar) | 53.73% | 77.05% | 5.896 | 2.519 |
|[ShuffleNetV2_x0_5](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_5_pretrained.tar) | 60.32% | 82.26% | 6.048 | 2.642 |
|[ShuffleNetV2_x1_5](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x1_5_pretrained.tar) | 71.63% | 90.15% | 6.113 | 3.164 |
|[ShuffleNetV2_x2_0](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x2_0_pretrained.tar) | 73.15% | 91.20% | 6.430 | 3.954 |
|[ShuffleNetV2_swish](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_swish_pretrained.tar) | 70.03% | 89.17% | 6.078 | 4.976 |

### AutoDL Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[DARTS_4M](https://paddle-imagenet-models-name.bj.bcebos.com/DARTS_GS_4M_pretrained.tar) | 75.23% | 92.15% | 13.572 | 6.335 |
|[DARTS_6M](https://paddle-imagenet-models-name.bj.bcebos.com/DARTS_GS_6M_pretrained.tar) | 76.03% | 92.79% | 16.406 | 6.864 |
- AutoDL基于可微结构搜索思路DARTS改进，引入Local Rademacher Complexity控制过拟合，并通过Resource Constraining灵活调整模型大小。

### ResNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[ResNet18](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar) | 70.98% | 89.92% | 3.456 | 2.261 |
|[ResNet18_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_vd_pretrained.tar) | 72.26% | 90.80% | 3.847 | 2.404 |
|[ResNet34](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar) | 74.57% | 92.14% | 5.668 | 3.424 |
|[ResNet34_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_vd_pretrained.tar) | 75.98% | 92.98% | 6.089 | 3.544 |
|[ResNet50](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar) | 76.50% | 93.00% | 8.787 | 5.137 |
|[ResNet50_vc](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vc_pretrained.tar) |78.35% | 94.03% | 9.013 | 5.285 |
|[ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar) | 79.12% | 94.44% | 9.058 | 5.259 |
|[ResNet50_vd_v2](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_v2_pretrained.tar)<sup>[1](#trans1)</sup> | 79.84% | 94.93% | 9.058 | 5.259 |
|[ResNet101](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) | 77.56% | 93.64% | 15.447 | 8.473 |
|[ResNet101_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar) | 80.17% | 94.97% | 15.685 | 8.574 |
|[ResNet152](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_pretrained.tar) | 78.26% | 93.96% | 21.816 | 11.646 |
|[ResNet152_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_vd_pretrained.tar) | 80.59% | 95.30% | 22.041 | 11.858 |
|[ResNet200_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet200_vd_pretrained.tar) | 80.93% | 95.33% | 28.015 | 14.896 |

<a name="trans1">[1]</a> 该预训练模型是在ResNet50_vd的预训练模型继续蒸馏得到的，用户可以通过ResNet50_vd的结构直接加载该预训练模型。

### Res2Net Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[Res2Net50_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net50_26w_4s_pretrained.tar) | 79.33% | 94.57% | 10.731 | 8.274 |
|[Res2Net50_vd_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net50_vd_26w_4s_pretrained.tar) | 79.75% | 94.91% | 11.012 | 8.493 |
|[Res2Net50_14w_8s](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net50_14w_8s_pretrained.tar) | 79.46% | 94.70% | 16.937 | 10.205 |
|[Res2Net101_vd_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net101_vd_26w_4s_pretrained.tar) | 80.64% | 95.22% | 19.612 | 14.651 |
|[Res2Net200_vd_26w_4s](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net200_vd_26w_4s_pretrained.tar) | 81.21% | 95.71% | 35.809 | 26.479 |

### ResNeXt Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_32x4d_pretrained.tar) | 77.75% | 93.82% | 12.863 | 9.241 |
|[ResNeXt50_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_vd_32x4d_pretrained.tar) | 79.56% | 94.62% | 13.673 | 9.162 |
|[ResNeXt50_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_64x4d_pretrained.tar) | 78.43% | 94.13% | 28.162 | 15.935 |
|[ResNeXt50_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_vd_64x4d_pretrained.tar) | 80.12% | 94.86% | 20.888 | 15.938 |
|[ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x4d_pretrained.tar) | 78.65% | 94.19% | 24.154 | 17.661 |
|[ResNeXt101_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_vd_32x4d_pretrained.tar) | 80.33% | 95.12% | 24.701 | 17.249 |
|[ResNeXt101_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_64x4d_pretrained.tar) | 78.35% | 94.52% | 41.073 | 31.288 |
|[ResNeXt101_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_vd_64x4d_pretrained.tar) | 80.78% | 95.20% | 42.277 | 32.620 |
|[ResNeXt152_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_32x4d_pretrained.tar) | 78.98% | 94.33% | 37.007 | 26.981 |
|[ResNeXt152_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_vd_32x4d_pretrained.tar) | 80.72% | 95.20% | 35.783 | 26.081 |
|[ResNeXt152_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_64x4d_pretrained.tar) | 79.51% | 94.71% | 58.966 | 47.915 |
|[ResNeXt152_vd_64x4d](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_vd_64x4d_pretrained.tar) | 81.08% | 95.34% | 60.947 | 47.406 |

### DenseNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[DenseNet121](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet121_pretrained.tar) | 75.66% | 92.58% | 12.437 | 5.592 |
|[DenseNet161](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet161_pretrained.tar) | 78.57% | 94.14% | 27.717 | 12.254 |
|[DenseNet169](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet169_pretrained.tar) | 76.81% | 93.31% | 18.941 | 7.742 |
|[DenseNet201](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet201_pretrained.tar) | 77.63% | 93.66% | 26.583 | 10.066 |
|[DenseNet264](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet264_pretrained.tar) | 77.96% | 93.85% | 41.495 | 14.740 |

### DPN Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[DPN68](https://paddle-imagenet-models-name.bj.bcebos.com/DPN68_pretrained.tar) | 76.78% | 93.43% | 18.446 | 6.199 |
|[DPN92](https://paddle-imagenet-models-name.bj.bcebos.com/DPN92_pretrained.tar) | 79.85% | 94.80% | 25.748 | 21.029 |
|[DPN98](https://paddle-imagenet-models-name.bj.bcebos.com/DPN98_pretrained.tar) | 80.59% | 95.10% | 29.421 | 13.411 |
|[DPN107](https://paddle-imagenet-models-name.bj.bcebos.com/DPN107_pretrained.tar) | 80.89% | 95.32% | 41.071 | 18.885 |
|[DPN131](https://paddle-imagenet-models-name.bj.bcebos.com/DPN131_pretrained.tar) | 80.70% | 95.14% | 41.179 | 18.246 |

### SENet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[SE_ResNet18_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet18_vd_pretrained.tar) | 73.33% | 91.38% | 4.715 | 3.061 |
|[SE_ResNet34_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet34_vd_pretrained.tar) | 76.51% | 93.20% | 7.475 | 4.299 |
|[SE_ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet50_vd_pretrained.tar) | 79.52% | 94.75% | 10.345 | 7.631 |
|[SE_ResNeXt50_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt50_32x4d_pretrained.tar) | 78.44% | 93.96% | 14.916 | 12.305 |
|[SE_ResNeXt50_vd_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt50_vd_32x4d_pretrained.tar) | 80.24% | 94.89% | 15.155 | 12.687 |
|[SE_ResNeXt101_32x4d](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt101_32x4d_pretrained.tar) | 79.12% | 94.20% | 30.085 | 23.218 |
|[SENet154_vd](https://paddle-imagenet-models-name.bj.bcebos.com/SENet154_vd_pretrained.tar) | 81.40% | 95.48% | 71.892 | 53.131 |

### Inception Series
| Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[GoogLeNet](https://paddle-imagenet-models-name.bj.bcebos.com/GoogLeNet_pretrained.tar) | 70.70% | 89.66% | 6.528 | 2.919 |
|[Xception41](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_pretrained.tar) | 79.30% | 94.53% | 13.757 | 7.885 |
|[Xception41_deeplab](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_deeplab_pretrained.tar) | 79.55% | 94.38% | 14.268 | 7.257 |
|[Xception65](https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_pretrained.tar) | 81.00% | 95.49% | 19.216 | 10.742 |
|[Xception65_deeplab](https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_deeplab_pretrained.tar) | 80.32% | 94.49% | 19.536 | 10.713 |
|[Xception71](https://paddle-imagenet-models-name.bj.bcebos.com/Xception71_pretrained.tar) | 81.11% | 95.45% | 23.291 | 12.154 |
|[InceptionV4](https://paddle-imagenet-models-name.bj.bcebos.com/InceptionV4_pretrained.tar) | 80.77% | 95.26% | 32.413 | 17.728 |

### DarkNet
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[DarkNet53](https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar) | 78.04% | 94.05% | 11.969 | 6.300 |

### ResNeXt101_wsl Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[ResNeXt101_32x8d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x8d_wsl_pretrained.tar) | 82.55% | 96.74% | 33.310 | 27.628 |
|[ResNeXt101_32x16d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x16d_wsl_pretrained.tar) | 84.24% | 97.26% | 54.320 | 47.599 |
|[ResNeXt101_32x32d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x32d_wsl_pretrained.tar) | 84.97% | 97.59% | 97.734 | 81.660 |
|[ResNeXt101_32x48d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x48d_wsl_pretrained.tar) | 85.37% | 97.69% | 161.722 |  |
|[Fix_ResNeXt101_32x48d_wsl](https://paddle-imagenet-models-name.bj.bcebos.com/Fix_ResNeXt101_32x48d_wsl_pretrained.tar) | 86.26% | 97.97% | 236.091 |  |

### EfficientNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[EfficientNetB0](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB0_pretrained.tar) | 77.38% | 93.31% | 10.303 | 4.334 |
|[EfficientNetB1](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB1_pretrained.tar)<sup>[2](#trans2)</sup> | 79.15% | 94.41% | 15.626 | 6.502 |
|[EfficientNetB2](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB2_pretrained.tar)<sup>[2](#trans2)</sup> | 79.85% | 94.74% | 17.847 | 7.558 |
|[EfficientNetB3](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB3_pretrained.tar)<sup>[2](#trans2)</sup> | 81.15% | 95.41% | 25.993 | 10.937 |
|[EfficientNetB4](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB4_pretrained.tar)<sup>[2](#trans2)</sup> | 82.85% | 96.23% | 47.734 | 18.536 |
|[EfficientNetB5](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB5_pretrained.tar)<sup>[2](#trans2)</sup> | 83.62% | 96.72% | 88.578 | 32.102 |
|[EfficientNetB6](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB6_pretrained.tar)<sup>[2](#trans2)</sup> | 84.00% | 96.88% | 138.670 | 51.059 |
|[EfficientNetB7](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB7_pretrained.tar)<sup>[2](#trans2)</sup> | 84.30% | 96.89% | 234.364 | 82.107 |
|[EfficientNetB0_small](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB0_Small_pretrained.tar)<sup>[3](#trans3)</sup> | 75.80% | 92.58% | 3.342 | 2.729 |

<a name="trans2">[2]</a> 表示该预训练权重是由[官方的代码仓库](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)转换来的。

<a name="trans3">[3]</a> 表示该预训练权重是在EfficientNetB0的基础上去除se模块，并使用通用的卷积训练的，精度稍稍下降，但是速度大幅提升。

### HRNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[HRNet_W18_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W18_C_pretrained.tar) | 76.92% | 93.39% | 23.013 | 11.601 |
|[HRNet_W30_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W30_C_pretrained.tar) | 78.04% | 94.02% | 25.793 | 14.367 |
|[HRNet_W32_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W32_C_pretrained.tar) | 78.28% | 94.24% | 29.564 | 14.328 |
|[HRNet_W40_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W40_C_pretrained.tar) | 78.77% | 94.47% | 33.880 | 17.616 |
|[HRNet_W44_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W44_C_pretrained.tar) | 79.00% | 94.51% | 36.021 | 18.990 |
|[HRNet_W48_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W48_C_pretrained.tar) | 78.95% | 94.42% | 30.064 | 19.963 |
|[HRNet_W64_C](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W64_C_pretrained.tar) | 79.30% | 94.61% | 38.921 | 24.742 |


### ResNet_ACNet Series
|Model | Top-1 | Top-5 | Paddle Fluid inference time(ms) | Paddle TensorRT inference time(ms) |
|- |:-: |:-: |:-: |:-: |
|[ResNet50_ACNet](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_ACNet_pretrained.tar)<sub>1</sub> | 76.71% | 93.24% | 13.205 | 8.804 |
|[ResNet50_ACNet](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_ACNet_deploy_pretrained.tar)<sub>2</sub> | 76.71% | 93.24% | 7.418 | 5.950 |

* 注:
    * `1`. 不对训练模型结果进行参数转换，进行评估。
    * `2`. 使用`sh ./utils/acnet/convert_model.sh`命令对训练模型结果进行参数转换，并设置`deploy mode=True`，进行评估。
    * `./utils/acnet/convert_model.sh`包含4个参数，分别是模型名称、输入的模型地址、输出的模型地址以及类别数量。

## FAQ

**Q:** 加载预训练模型报错，Enforce failed. Expected x_dims[1] == labels_dims[1], but received x_dims[1]:1000 != labels_dims[1]:6.

**A:** 类别数匹配不上，删掉最后一层分类层FC

**Q:** reader中报错AttributeError: 'NoneType' object has no attribute 'shape'

**A:** 文件路径load错误

**Q:** 出现cudaStreamSynchronize an illegal memory access was encountered errno:77 错误

**A:** 可能是因为显存问题导致，添加如下环境变量：

    export FLAGS_fast_eager_deletion_mode=1
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98

## 参考文献
- AlexNet: [imagenet-classification-with-deep-convolutional-neural-networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- ResNet: [Deep Residual Learning for Image Recognitio](https://arxiv.org/abs/1512.03385), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- ResNeXt: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431), Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
- SeResNeXt: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)Jie Hu, Li Shen, Samuel Albanie
- ShuffleNetV1: [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083), Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
- ShuffleNetV2: [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164), Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
- MobileNetV1: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861), Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
- MobileNetV2: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381v4.pdf), Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
- MobileNetV3: [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf), Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
- VGG: [Very Deep Convolutional Networks for Large-scale Image Recognition](https://arxiv.org/pdf/1409.1556), Karen Simonyan, Andrew Zisserman
- GoogLeNet: [Going Deeper with Convolutions](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf), Christian Szegedy1, Wei Liu2, Yangqing Jia
- Xception: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357), Franc ̧ois Chollet
- InceptionV4: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261), Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
- DarkNet: [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), Joseph Redmon, Ali Farhadi
- DenseNet: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993), Gao Huang, Zhuang Liu, Laurens van der Maaten
- DPN: [Dual Path Networks](https://arxiv.org/pdf/1707.01629.pdf), Yunpeng Chen, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, Jiashi Feng
- SqueezeNet: [SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE](https://arxiv.org/abs/1602.07360), Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
- ResNeXt101_wsl: [Exploring the Limits of Weakly Supervised Pretraining](https://arxiv.org/abs/1805.00932), Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, Laurens van der Maaten
- Fix_ResNeXt101_wsl: [Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423), Hugo Touvron, Andrea Vedaldi, Matthijs Douze, Herve ́ Je ́gou
- EfficientNet: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946), Mingxing Tan, Quoc V. Le
- Res2Net: [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169), Shang-Hua Gao, Ming-Ming Cheng, Kai Zhao, Xin-Yu Zhang, Ming-Hsuan Yang, Philip Torr
- HRNet: [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919), Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao
- DARTS: [DARTS: Differentiable Architecture Search](https://arxiv.org/pdf/1806.09055.pdf), Hanxiao Liu, Karen Simonyan, Yiming Yang
- ACNet: [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](https://arxiv.org/abs/1908.03930), Xiaohan Ding, Yuchen Guo, Guiguang Ding, Jungong Han

## 版本更新
- 2018/12/03 **Stage1**: 更新AlexNet，ResNet50，ResNet101，MobileNetV1
- 2018/12/23 **Stage2**: 更新VGG系列，SeResNeXt50_32x4d，SeResNeXt101_32x4d，ResNet152
- 2019/01/31 更新MobileNetV2_x1_0
- 2019/04/01 **Stage3**: 更新ResNet18，ResNet34，GoogLeNet，ShuffleNetV2
- 2019/06/12 **Stage4**: 更新ResNet50_vc，ResNet50_vd，ResNet101_vd，ResNet152_vd，ResNet200_vd，SE154_vd InceptionV4，ResNeXt101_64x4d，ResNeXt101_vd_64x4d
- 2019/06/22 更新ResNet50_vd_v2
- 2019/07/02 **Stage5**: 更新MobileNetV2_x0_5，ResNeXt50_32x4d，ResNeXt50_64x4d，Xception41，ResNet101_vd
- 2019/07/19 **Stage6**: 更新ShuffleNetV2_x0_25，ShuffleNetV2_x0_33，ShuffleNetV2_x0_5，ShuffleNetV2_x1_0，ShuffleNetV2_x1_5，ShuffleNetV2_x2_0，MobileNetV2_x0_25，MobileNetV2_x1_5，MobileNetV2_x2_0，ResNeXt50_vd_64x4d，ResNeXt101_32x4d，ResNeXt152_32x4d
- 2019/08/01 **Stage7**: 更新DarkNet53，DenseNet121，Densenet161，DenseNet169，DenseNet201，DenseNet264，SqueezeNet1_0，SqueezeNet1_1，ResNeXt50_vd_32x4d，ResNeXt152_64x4d，ResNeXt101_32x8d_wsl，ResNeXt101_32x16d_wsl，ResNeXt101_32x32d_wsl，ResNeXt101_32x48d_wsl，Fix_ResNeXt101_32x48d_wsl
- 2019/09/11 **Stage8**: 更新ResNet18_vd，ResNet34_vd，MobileNetV1_x0_25，MobileNetV1_x0_5，MobileNetV1_x0_75，MobileNetV2_x0_75，MobilenNetV3_small_x1_0，DPN68，DPN92，DPN98，DPN107，DPN131，ResNeXt101_vd_32x4d，ResNeXt152_vd_64x4d，Xception65，Xception71，Xception41_deeplab，Xception65_deeplab，SE_ResNet50_vd
- 2019/09/20 更新EfficientNet
- 2019/11/28 **Stage9**: 更新SE_ResNet18_vd，SE_ResNet34_vd，SE_ResNeXt50_vd_32x4d，ResNeXt152_vd_32x4d，Res2Net50_26w_4s，Res2Net50_14w_8s，Res2Net50_vd_26w_4s，HRNet_W18_C，HRNet_W30_C，HRNet_W32_C，HRNet_W40_C，HRNet_W44_C，HRNet_W48_C，HRNet_W64_C
- 2020/01/07 **Stage10**: 更新AutoDL Series
- 2020/01/09 **Stage11**: 更新Res2Net101_vd_26w_4s, Res2Net200_vd_26w_4s

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
