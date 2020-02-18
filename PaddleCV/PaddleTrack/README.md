# PaddleTrack 单目标跟踪框架

## 目标跟踪介绍

PaddleTrack是基于百度深度学习框架Paddle研发的视频单目标跟踪（Visual Object Tracking）库。涵盖当前目标跟踪的主流模型，包括SiamFC, SiamRPN, SiamMask, ATOM。PaddleTrack旨在给开发者提供一系列基于PaddlePaddle的便捷、高效的目标跟踪深度学习算法，后续会不断的扩展模型的丰富度。


## 开始使用

### 数据准备

目标跟踪的训练集和测试集是不同的，目前最好的模型往往是使用多个训练集进行训练。常用的数据集如下:


主流的训练数据集有：
- [VID](http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz)
- [Microsoft COCO 2014](http://cocodataset.org/#download)
- [LaSOT](https://drive.google.com/file/d/1O2DLxPP8M4Pn4-XCttCJUW3A29tDIeNa/view)

主流跟踪评估数据集有：
- [OTB](https://paddlemodels.bj.bcebos.com/paddle_track/vot/OTB100.tar)
- [VOT](https://paddlemodels.bj.bcebos.com/paddle_track/vot/VOT2015.tar)
- [LaSOT](https://drive.google.com/file/d/1O2DLxPP8M4Pn4-XCttCJUW3A29tDIeNa/view)
- [GOT-10K](http://got-10k.aitestunion.com/downloads)

数据评估将主要在VOT上完成，我们提供了OTB和VOT的数据下载链接。


## 快速开始

### 训练准备，安装第三方库

```bash
# (可选) 0. 强烈建议新建一个 conda 环境，在安装 anaconda 后执行
#      conda create -n paddle1.7-py3.6 python=3.6
#      conda activate paddle1.7-py3.6

# 1. 安装依赖库
# 神经网络计算
pip install paddlepaddle_gpu==1.7

# 训练数据多线程载入
pip install --upgrade git+https://github.com/tensorpack/dataflow.git

# 使用基于 C 语言的拓展程序
pip install cython

# coco 数据 API
pip install pycocotools

# lmdb 数据库
pip install lmdb

# csv 处理
pip install pandas

# (可选) 推荐安装：快速读取 jpeg 文件
apt-get install libturbojpeg

# jpeg 读取
pip install jpeg4py

# 视觉计算
pip install opencv-python

# 训练结果可视化
pip install tensorboardX

# 跟踪结果可视化
pip install videofig

# jupyter notebook
pip install jupyter

# 进度条
pip install tqdm

# (可选) 推荐安装：进程控制
apt-get install build-essential libcap-dev
pip install python-prctl

# 2. 安装 paddle 跟踪框架
# 下载代码库
# TODO: change to real repo
git clone PaddleTrack

# 编译
cd PaddleTrack/pytracking_pp/pysot_toolkit/utils
python setup.py build_ext --inplace
```


### 预训练 backbone 下载

在开始训练前，先准备SiamRPN、SiamMask、ATOM模型的Backbone预训练模型。

我们提供 ATOM ResNet18 和 ResNet50 的 backbone模型。可从[这里](https://paddlemodels.bj.bcebos.com/paddle_track/vot/pretrained_models.tar)下载所有预训练模型的压缩包。下载压缩包后，解压后backbone文件夹下的模型为ResNet18和ResNet50模型的预训练模型。
压缩包解压后的文件夹为 `pretrained_models`. 文件的目录结构如下：

```
pretrained_models
    └─ atom
        └─ atom_resnet18.pdparams
        └─ atom_resnet50.pdparams
        ...
    └─ atom-torch
        └─ atom_default.pth
        ...
    └─ backbone
        └─ ResNet18.pdparams
        └─ ResNet50.pdparams
```


### 设置训练环境

```bash
# 到代码库根目录
cd PaddleTrack

# 生成 local.py 文件，再次训练时不需要重新生成
python -c "from ltr_pp.admin.environment import create_default_local_file; create_default_local_file()"

# 用你常用的编辑器编辑 ltr_pp/admin/local.py
# 比方说，vim ltr_pp/admin/local.py
# 其中，
#       workspace_dir = './checkpoints' # 要保存训练模型的位置
#       backbone_dir = Your BACKBONE_PATH # 训练SiamFC时不需要设置
#       并依次设定需要使用的训练数据集如 VID, LaSOT, COCO 等，比如：
#       imagenet_dir = '/Datasets/ILSVRC2015/'  # 设置训练集VID的路径
```
训练SiamFC时需要只需要配置 workspace_dir和 imagenet_dir即可，如下：
```bash
    self.workspace_dir = './checkpoints'
    self.imagenet_dir = '/Datasets/ILSVRC2015/'
```


### 开始训练

```bash
# 到训练代码目录
cd ltr_pp

# 训练 ATOM ResNet18
python run_training.py bbreg atom_res18_vid_lasot_coco

# 训练 ATOM ResNet50
python run_training.py bbreg atom_res50_vid_lasot_coco

# 训练 SiamFC
python run_training.py siamfc siamfc_alexnet_vid
```


### 评测自己训练模型

接下来开始设置评估环境：
```bash
# 生成 local.py 文件
python -c "from pytracking_pp.admin.environment import create_default_local_file; create_default_local_file()"

# 用你常用的编辑器编辑 pytracking_pp/pysot_toolkit/local.py
# 比方说，vim pytracking_pp/pysot_toolkit/local.py
# 其中 dataset_path 和 network_path 设置为上述的 DATASET_PATH 和 NETWORK_PATH
```


将自己训练的模型拷贝到 `NETWORK_PATH`,或者建立软链接，如
```bash
ln -s PaddleTrack/ltr_pp/Logs/checkpoints/ltr_pp/bbreg/atom_res18_vid_lasot_coco $NETWORK_PATH/bbreg
```

#### 开始测试：

测试ATOM模型：
```bash
# 在VOT2018上评测ATOM模型
# -d VOT2018  表示使用VOT2018数据集进行评测
# -tr bbreg.atom_res18_vid_lasot_coco 表示要评测的模型，和训练保持一致
# -te atom.default_vot 表示加载定义超参数的文件pytracking_pp/parameter/atom/default_vot.py 
# -e 40 表示使用第40个epoch的模型进行评测
cd pytracking_pp
python eval_benchmark.py -d VOT2018 -tr bbreg.atom_res18_vid_lasot_coco -te atom.default_vot -e 40 
```


测试SiamFC
```

# 在 OTB2013 上评测 SiamFC
cd pytracking_pp/tracker/siamfc
python eval_siamfc_otb.py --checkpoint "your trained params path" --dataset_dir "your test dataset path" --dataset_name "test dataset name" --start_epoch 1 --end_epoch 50

# 例如，在OTB2013上测试SiamFC
python eval_siamfc_otb.py --checkpoint "/checkpoints/ltr_pp/siamfc/siamfc_alexnet_vid/" --dataset_dir "/Datasets/OTB100/" --dataset_name 'CVPR13' --start_epoch 12 --end_epoch 50

# 例如，在VOT15上测试SiamFC
python eval_siamfc_vot.py --checkpoint "/checkpoints/ltr_pp/siamfc/siamfc_alexnet_vid/" --dataset_dir "/Datasets/VOT2015/" --dataset_name 'VOT2015' --start_epoch 12 --end_epoch 50
```



## 跟踪结果可视化

在数据集上评测完后，可以通过可视化跟踪器的结果来定位问题。我们提供下面的方法来可视化跟踪结果：
```bash
cd pytracking_pp

# 开启 jupyter notebook，请留意终端是否输出 token
jupyter notebook --ip 0.0.0.0 --port 8888
```

在你的浏览器中输入服务器的 IP 地址加上端口号，若是在本地执行则打开
`http://localhost:8888`。若需要输入 token 请查看执行 `jupyter notebook --ip 0.0.0.0 --port 8888` 命令时的终端输出。

打开网页之后，打开 `visualize_results_on_benchmark.ipynb` 来可视化结果。

## 指标结果

| 数据集 | 模型 | Backbone | 论文结果 | 训练结果 | 模型|
| :-------: | :-------: | :---: | :---: | :---------: |:---------: |
|OTB2013| SiamFC | Alexnet |  AUC（OPE）：60.8  | 61.8 | [model]() |
|VOT2015| SiamFC | Alexnet |  ACC & Failure：0.5335 & 84  | 0.5415 & 84 | [model]() |
|VOT2018| ATOM | Res18 |  EAO: 0.401 | 0.399 | [model]() |

