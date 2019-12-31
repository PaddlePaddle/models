# [Dual Attention Network for Scene Segmentation (CVPR2019)](https://arxiv.org/pdf/1809.02983.pdf)

本项目是[DANet](https://arxiv.org/pdf/1809.02983.pdf)的 PaddlePaddle（>=1.5.2） 实现， 包含模型训练，验证等内容。

## 模型简介
![net](img/Network.png)
骨干网络使用ResNet，为更好地进行语义分割任务，作者对ResNet做出以下改动：

    1、将最后两个layer的downsampling取消，使得特征图是原图的1/8，保持较高空间分辨率。
    2、最后两个layer采用空洞卷积扩大感受野。
然后接上两个并行的注意力模块（位置注意力和通道注意力），最终将两个模块的结果进行elementwise操作，之后再接一层卷积输出分割图。

### 位置注意力

![position](img/position.png)

A是骨干网络ResNet输出经过一层卷积生成的特征图，维度为CHW；
A经过3个卷积操作输出维度均为CHW的B、C、D。将B、C、D都reshape到CN（N = H*W）；
然后将B reshape后的结果转置与C相乘，得到N * N的矩阵， 对于矩阵的每一个点进行softmax；
然后将D与softmax后的结果相乘并reshape到CHW，再与A进行elementwise。

### 通道注意力
![channel](img/channel.png)


A是骨干网络ResNet输出经过一层卷积生成的特征图，维度为CHW；
A经过3个reshape操作输出维度均为CN（N = H*W）的B、C、D；
然后将B转置与C相乘，得到C * C的矩阵，对于矩阵的每一个点进行softmax；
然后将D与softmax后的结果相乘并reshape到CHW，再与A进行elementwise。



## 数据准备

公开数据集：Cityscapes

训练集2975张，验证集500张，测试集1525张，图片分辨率都是1024*2048。

数据集来源：AIstudio数据集页面上[下载](https://aistudio.baidu.com/aistudio/datasetDetail/11503),  cityscapes.zip解压至dataset文件夹下,train.zip解压缩到cityscapes/leftImg8bit，其目录结构如下：
```text
dataset
  ├── cityscapes               # Cityscapes数据集
         ├── gtFine            # 精细化标注的label
         ├── leftImg8bit       # 训练，验证，测试图片
         ├── trainLabels.txt   # 训练图片路径
         ├── valLabels.txt     # 验证图片路径
              ...               ...
```
## 训练说明

#### 数据增强策略
    1、随机尺度缩放：尺度范围0.75到2.0
    2、随机左右翻转：发生概率0.5
    3、同比例缩放：缩放的大小由选项1决定。
    4、随机裁剪：
    5、高斯模糊：发生概率0.3（可选）
    6、颜色抖动，对比度，锐度，亮度; 发生概率0.3（可选）
###### 默认1、2、3、4、5、6都开启

#### 学习率调节策略
    1、使用热身策略，学习率由0递增到base_lr，热身轮数（epoch）是5
    2、在热身策略之后使用学习率衰减策略（poly），学习率由base_lr递减到0

#### 优化器选择
	Momentum: 动量0.9，正则化系数1e-4

#### 加载预训练模型
	设置 --load_pretrained_model（默认为False）
	预训练文件：
	    checkpoint/DANet50_pretrained_model_paddle1.6.pdparams
        checkpoint/DANet101_pretrained_model_paddle1.6.pdparams

#### 加载训练好的模型
	设置 --load_better_model（默认为False）
	训练好的文件：
		checkpoint/DANet101_better_model_paddle1.6.pdparams
##### 【注】
    训练时paddle版本是1.5.2，代码已转为1.6版本（兼容1.6版本），预训练参数、训练好的参数来自1.5.2版本

#### 配置模型文件路径
[预训练参数、最优模型参数下载](https://paddlemodels.bj.bcebos.com/DANet/DANet_models.tar)

其目录结构如下：
```text
checkpoint
    ├── DANet50_pretrained_model_paddle1.6.pdparams       # DANet50预训练模型，需要paddle >=1.6.0
    ├── DANet101_pretrained_model_paddle1.6.pdparams      # DANet101预训练模型，需要paddle >=1.6.0
    ├── DANet101_better_model_paddle1.6.pdparams          # DANet101训练最优模型，需要paddle >=1.6.0
    ├── DANet101_better_model_paddle1.5.2                 # DANet101在1.5.2版本训练的最优模型，需要paddle >= 1.5.2

```

## 模型训练

```sh
cd danet
export PYTHONPATH=`pwd`:$PYTHONPATH
# open garbage collection to save memory
export FLAGS_eager_delete_tensor_gb=0.0
# setting visible devices for train
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

executor执行以下命令进行训练
```sh
python train_executor.py --backbone resnet101 --base_size 1024 --crop_size 768 --epoch_num 350 --batch_size 2 --lr 0.003 --lr_scheduler poly --warm_up --warmup_epoch 2 --cuda --use_data_parallel --load_pretrained_model --save_model checkpoint/DANet101_better_model_paddle1.5.2 --multi_scales --flip --dilated --multi_grid --scale --multi_dilation 4 8 16
```
参数含义： 使用ResNet101骨干网络，训练图片基础大小是1024，裁剪大小是768，训练轮数是350次，batch size是2
学习率是0.003，学习率衰减策略是poly，使用学习率热身，热身轮数是2轮，使用GPU，使用数据并行， 加载预训练模型，设置加载的模型地址，使用多尺度测试， 使用图片左右翻转测试，使用空洞卷积，使用multi_grid，multi_dilation设置为4 8 16，使用多尺度训练
##### Windows下训练需要去掉 --use_data_parallel
#### 或者
dygraph执行以下命令进行训练
```sh
python train_dygraph.py --backbone resnet101 --base_size 1024 --crop_size 768 --epoch_num 350 --batch_size 2 --lr 0.003 --lr_scheduler poly --cuda --use_data_parallel --load_pretrained_model --save_model checkpoint/DANet101_better_model_paddle1.6 --multi_scales --flip --dilated --multi_grid --scale --multi_dilation 4 8 16
```
参数含义： 使用ResNet101骨干网络，训练图片基础大小是1024，裁剪大小是768，训练轮数是350次，batch size是2，学习率是0.003，学习率衰减策略是poly，使用GPU， 使用数据并行，加载预训练模型，设置加载的模型地址，使用多尺度测试，使用图片左右翻转测试，使用空洞卷积，使用multi_grid，multi_dilation设置4 8 16，使用多尺度训练

#### 【注】
##### train_executor.py使用executor方式训练（适合paddle >= 1.5.2），train_dygraph.py使用动态图方式训练（适合paddle >= 1.6.0），两种方式都可以
##### 动态图方式训练暂时不支持学习率热身

#### 在训练阶段，输出的验证结果不是真实的，需要使用eval.py来获得验证的最终结果。

 ## 模型验证
```sh
# open garbage collection to save memory
export FLAGS_eager_delete_tensor_gb=0.0
# setting visible devices for prediction
export CUDA_VISIBLE_DEVICES=0

python eval.py --backbone resnet101 --base_size 2048 --crop_size 1024 --cuda --use_data_parallel --load_better_model --save_model checkpoint/DANet101_better_model_paddle1.6 --multi_scales --flip --dilated --multi_grid --multi_dilation 4 8 16
```
##### 如果需要把executor训练的参数转成dygraph模式下进行验证的话，请在命令行加上--change_executor_to_dygraph

## 验证结果
评测指标：mean IOU(平均交并比)


| 模型 | 单尺度 | 多尺度 |
| :---:|:---:| :---:|
|DANet101|0.8043836|0.8138021

##### 具体数值
| 模型 | cls1 | cls2 | cls3 | cls4 | cls5 | cls6 | cls7 | cls8 | cls9 | cls10 | cls11 | cls12 | cls13 | cls14 | cls15 | cls16 |cls17 | cls18 | cls19 |
| :---:|:---: | :---:| :---:|:---: | :---:| :---:|:---: | :---:| :---:|:---:  |:---: |:---:  |:---:  | :---: | :---: |:---:  | :---:| :---: |:---:  |
|DANet101-SS|0.98212|0.85372|0.92799|0.59976|0.63318|0.65819|0.72023|0.80000|0.92605|0.65788|0.94841|0.83377|0.65206|0.95566|0.87148|0.91233|0.84352|0.71948|0.78737|
|DANet101-MS|0.98047|0.84637|0.93084|0.62699|0.64839|0.67769|0.73650|0.81343|0.92942|0.67010|0.95127|0.84466|0.66635|0.95749|0.87755|0.92370|0.85344|0.73007|0.79742|

## 输出结果可视化
![val_1](img/val_1.png)
###### 输入图片
![val_gt](img/val_gt.png)
###### 图片label
![val_output](img/val_output.png)
###### DANet101模型输出
