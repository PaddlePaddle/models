# tracking 单目标跟踪框架

## 介绍

tracking 是基于百度深度学习框架Paddle研发的视频单目标跟踪（Visual Object Tracking, VOT）库, 整体框架参考 [pytracking](https://github.com/visionml/pytracking)，其优秀的设计使得我们能够方便地将其他跟踪器如SiamFC，SiamRPN，SiamMask等融合到一个框架中，方便后续统一的实验和比较。

当前tracking涵盖当前目标跟踪的主流模型，包括SiamFC, SiamRPN, SiamMask, ATOM。tracking旨在给开发者提供一系列基于PaddlePaddle的便捷、高效的目标跟踪深度学习算法，后续会不断的扩展模型的丰富度。

ATOM 跟踪效果展示：

![ball](./imgs/ball1.gif)

图中，绿色框为标注的bbox，红色框为ATOM跟踪的bbox。

## 代码目录结构


```
imgs 包含跟踪结果的图像

ltr 包含模型训练代码
  └─ actors             输入数据，输出优化目标  
  └─ admin              管理数据路径等
  └─ data               多线程数据读取和预处理
  └─ dataset            训练数据集读取
  └─ models             模型定义
  └─ train_settings     训练配置
  └─ trainers           模型训练器
  └─ run_training.py    模型训练入口程序

pytracking  包含跟踪代码
  └─ admin              管理数据路径，模型位置等
  └─ features           特征提取
  └─ libs               跟踪常用操作
  └─ parameter          跟踪器参数设置
  └─ tracker            跟踪器
  └─ utils              画图等
  └─ pysot-toolkit      评估数据集载入和指标计算
  └─ eval_benchmark.py  评估跟踪器入口程序
  └─ visualize_results_on_benchmark.ipynb  可视化跟踪结果
```

## 开始使用

### 数据准备

目标跟踪的训练集和测试集是不同的，目前最好的模型往往是使用多个训练集进行训练。

主流的训练数据集有：
- [VID](http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz)
- [Microsoft COCO 2014](http://cocodataset.org/#download)
- [LaSOT](https://drive.google.com/file/d/1O2DLxPP8M4Pn4-XCttCJUW3A29tDIeNa/view)
- [GOT-10K](http://got-10k.aitestunion.com/downloads_dataset/full_data)

下载并解压后的数据集的组织方式为：
```
/Datasets/
    └─ ILSVRC2015_VID/
    └─ train2014/
    └─ GOT-10K/
    └─ LaSOTBenchmark/

```
Datasets是数据集保存的路径。

注：数据集较大，请预留足够的磁盘空间。训练Siamfc时，只需要下载VID数据集，训练ATOM需要全部下载上述三个数据集。


## 快速开始

tracking的工作环境：
- Linux
- python3
- PaddlePaddle1.7

> 注意：如果遇到cmath无法import的问题，建议切换Python版本，建议使用python3.6.8, python3.7.0 。另外，
> tracking暂不支持在window上运行，如果开发者有需求在window上运行tracking，请在issue中提出需求。

### 安装依赖

1. 安装paddle，需要安装1.7版本的Paddle，如低于这个版本，请升级到Paddle 1.7.
```bash
pip install paddlepaddle-gpu==1.7.0
```

2. 安装第三方库，建议使用anaconda
```bash
# (可选) 0. 强烈建议新建一个 conda 环境，在安装 anaconda 后执行
#      conda create -n paddle1.7-py3.6 python=3.6
#      conda activate paddle1.7-py3.6

cd tracking
pip install -r requirements.txt

# (可选) 1. 推荐安装：快速读取 jpeg 文件
apt-get install libturbojpeg

# (可选) 2. 推荐安装：进程控制
apt-get install build-essential libcap-dev
pip install python-prctl
```



### 预训练 backbone 下载

在开始训练前，先准备SiamRPN、SiamMask、ATOM模型的Backbone预训练模型。

我们提供 ATOM ResNet18 和 ResNet50 的 backbone模型。可从[这里](https://paddlemodels.bj.bcebos.com/paddle_track/vot/pretrained_models.tar)下载所有预训练模型的压缩包。
压缩包解压后的文件夹为 `pretrained_models`. 文件的目录结构如下：
```
/pretrained_models/
    └─ atom
        └─ atom_resnet18.pdparams
        └─ atom_resnet50.pdparams
    └─ backbone
        └─ ResNet18.pdparams
        └─ ResNet50.pdparams
```
其中/pretrained_models/backbone/文件夹包含，ResNet18、ResNet50在Imagenet上的预训练模型。


### 设置训练参数

在启动训练前，需要设置tracking使用的数据集路径，以及训练模型保存的路径，这些参数在ltr/admin/local.py中设置。

首先，需要先生成local.py文件。

```bash
# 到代码库根目录
cd tracking

```
其次，设置训练模型文件保存路径：workspace_dir，backbone模型路径：backbone_dir，数据集路径等等，对于没有用到的数据集，可以不用设置其路径。
```
# 用你常用的编辑器编辑 ltr/admin/local.py
# 比方说，vim ltr/admin/local.py
# 其中，
#       workspace_dir = './checkpoints' # 要保存训练模型的位置
#       backbone_dir = Your BACKBONE_PATH # 训练SiamFC时不需要设置
#       并依次设定需要使用的训练数据集如 VID, LaSOT, COCO 等，比如：
#       imagenet_dir = '/Datasets/ILSVRC2015/'  # 设置训练集VID的路径

# 如果 ltr/admin/local.py 不存在，请使用代码生成
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

训练SiamFC时需要只需要配置 workspace_dir和 imagenet_dir即可，如下：
```bash
    self.workspace_dir = './checkpoints'
    self.imagenet_dir = '/Datasets/ILSVRC2015/'
```
训练ATOM时，除了 workspace_dir和 imagenet_dir外，还需要指定coco, lasot, got10k的数据集路径，参考如下：
```bash
    self.workspace_dir = './checkpoints'
    self.lasot_dir = '/Datasets/LaSOTBenchmark/'
    self.coco_dir = '/Datasets/train2014/'
    self.got10k_dir = '/Datasets/GOT-10k/train'
    self.imagenet_dir = '/Datasets/ILSVRC2015/'
```
另外，训练ATOM时，需要准备got10k和lasot的数据集划分文件，方式如下：
```bash
cd ltr/data_specs/
wget https://paddlemodels.cdn.bcebos.com/paddle_track/vot/got10k_lasot_split.tar
tar xvf got10k_lasot_split.tar
```


### 启动训练

```bash
# 到训练代码目录
cd ltr

# 训练 ATOM ResNet18
python run_training.py bbreg atom_res18_vid_lasot_coco

# 训练 ATOM ResNet50
python run_training.py bbreg atom_res50_vid_lasot_coco

# 训练 SiamFC
python run_training.py siamfc siamfc_alexnet_vid
```


## 模型评估

评估训练后的模型使用[pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit)工具包，其提供了多个单目标跟踪数据集的评估API。测试数据集建议从pysot-toolkit 提供的链接中下载。

准备好测试数据后，使用如下命令，克隆跟踪评估pysot-toolkit的代码模块，运行如下命令：

```bash
cd pytracking
git clone https://github.com/StrangerZhang/pysot-toolkit.git
mv pysot-toolkit pysot_toolkit
cd pysot_toolkit
pip install -r requirements.txt
cd pysot/utils/
python setup.py build_ext --inplace
```

### 测试数据集准备
按照pysot-toolkit的方式准备数据集VOT2018，放到/Datasets 文件夹下。

### 设置模型评估环境
接下来开始设置评估环境：
```bash
# 在pytracking/admin/local.py文件中设置测试数据集、待测试模型、以及测试结果的保存路径
# 用你常用的编辑器编辑 pytracking/admin/local.py
# 比方说，vim pytracking/admin/local.py
# 其中 settings.dataset_path 和 settings.network_path 分别设置为测试集的路径和模型训练参数的路径

# 如果不存在 pytracking/admin/local.py，可以使用代码生成
python -c "from pytracking.admin.environment import create_default_local_file; create_default_local_file()"
```

### 准备测试数据和模型
按照pysot-toolkit的方式准备数据集VOT2018，放到settings.dataset_path指定文件夹中，或者自行设置settings.dataset_path指向测试数据集。


将自己训练的模型拷贝到 `NETWORK_PATH`，或者建立软链接，如
```bash
ln -s tracking/ltr/Logs/checkpoints/ltr/bbreg/ $NETWORK_PATH/bbreg
```

### 开始测试：

测试ATOM模型：
```bash
# 在VOT2018上评测ATOM模型
# -d VOT2018  表示使用VOT2018数据集进行评测
# -tr bbreg.atom_res18_vid_lasot_coco 表示要评测的模型，和训练保持一致
# -te atom.default_vot 表示加载定义超参数的文件pytracking/parameter/atom/default_vot.py
# -e 40 表示使用第40个epoch的模型进行评测，也可以设置为'range(1, 50, 1)' 表示测试从第1个epoch到第50个epoch模型
# -n 15 表示测试15次取平均结果，默认值是1
python eval_benchmark.py -d VOT2018 -tr bbreg.atom_res18_vid_lasot_coco -te atom.default_vot -e 40 -n 15
```

测试SiamFC
```
# 在VOT2018上测试SiamFC
python eval_benchmark.py -d VOT2018 -tr siamfc.siamfc_alexnet_vid -te siamfc.default -e 'range(1, 50, 1)'
```



## 跟踪结果可视化


在数据集上评测完后，可以通过可视化跟踪器的结果来定位问题。我们提供下面的方法来可视化跟踪结果：
```bash
cd pytracking

# 开启 jupyter notebook，请留意终端是否输出 token
jupyter notebook --ip 0.0.0.0 --port 8888
```

在你的浏览器中输入服务器的 IP 地址加上端口号，若是在本地执行则打开
`http://localhost:8888`。若需要输入 token 请查看执行 `jupyter notebook --ip 0.0.0.0 --port 8888` 命令时的终端输出。

打开网页之后，打开 `visualize_results_on_benchmark.ipynb` 来可视化结果。

## 指标结果

| 数据集 | 模型 | Backbone | 论文结果 | 训练结果 | 模型|
| :-------: | :-------: | :---: | :---: | :---------: |:---------: |
|VOT2018| ATOM | Res18 |  EAO: 0.401 | 0.399 | [model](https://paddlemodels.cdn.bcebos.com/paddle_track/vot/ATOM.tar) |
|VOT2018| SiamFC | AlexNet |  EAO: 0.188 | 0.211 | [model](https://paddlemodels.cdn.bcebos.com/paddle_track/vot/SiamFC.tar) |

## 引用与参考

SiamFC **[[Paper]](https://arxiv.org/pdf/1811.07628.pdf) [[Code]](https://www.robots.ox.ac.uk/~luca/siamese-fc.html)**

    @inproceedings{bertinetto2016fully,
      title={Fully-convolutional siamese networks for object tracking},
      author={Bertinetto, Luca and Valmadre, Jack and Henriques, Joao F and Vedaldi, Andrea and Torr, Philip HS},
      booktitle={European conference on computer vision},
      pages={850--865},
      year={2016},
      organization={Springer}
    }

ATOM **[[Paper]](https://arxiv.org/pdf/1811.07628.pdf)  [[Raw results]](https://drive.google.com/drive/folders/1MdJtsgr34iJesAgL7Y_VelP8RvQm_IG_) [[Models]](https://drive.google.com/open?id=1EsNSQr25qfXHYLqjZaVZElbGdUg-nyzd)  [[Training Code]](https://github.com/visionml/pytracking/blob/master/ltr/README.md#ATOM)  [[Tracker Code]](https://github.com/visionml/pytracking/blob/master/pytracking/README.md#ATOM)**  

    @inproceedings{danelljan2019atom,
      title={Atom: Accurate tracking by overlap maximization},
      author={Danelljan, Martin and Bhat, Goutam and Khan, Fahad Shahbaz and Felsberg, Michael},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={4660--4669},
      year={2019}
    }

DiMP **[[Paper]](https://arxiv.org/pdf/1904.07220v1.pdf)  [[Raw results]](https://drive.google.com/drive/folders/15mpUAJmzxemnOC6gmvMTCDJ-0v6hxJ7y) [[Models]](https://drive.google.com/open?id=1YEJySjhFokyQ6zgQg6vFAnzEFi1Onq7G)  [[Training Code]](https://github.com/visionml/pytracking/blob/master/ltr/README.md#DiMP)  [[Tracker Code]](https://github.com/visionml/pytracking/blob/master/pytracking/README.md#DiMP)**  

    @inproceedings{bhat2019learning,
      title={Learning discriminative model prediction for tracking},
      author={Bhat, Goutam and Danelljan, Martin and Gool, Luc Van and Timofte, Radu},
      booktitle={Proceedings of the IEEE International Conference on Computer Vision},
      pages={6182--6191},
      year={2019}
    }

ECO **[[Paper]](https://arxiv.org/pdf/1611.09224.pdf)  [[Models]](https://drive.google.com/open?id=1aWC4waLv_te-BULoy0k-n_zS-ONms21S)  [[Tracker Code]](https://github.com/visionml/pytracking/blob/master/pytracking/README.md#ECO)**  

    @inproceedings{danelljan2017eco,
      title={Eco: Efficient convolution operators for tracking},
      author={Danelljan, Martin and Bhat, Goutam and Shahbaz Khan, Fahad and Felsberg, Michael},
      booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages={6638--6646},
      year={2017}
    }
