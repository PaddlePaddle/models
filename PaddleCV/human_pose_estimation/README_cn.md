# 关键点检测（Simple Baselines for Human Pose Estimation）

## 介绍
本目录包含了对论文[Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208) (ECCV'18)的复现.

![demo](demo.gif)

> **演示视频**: *Bruno Mars - That’s What I Like [官方视频]*.

同时推荐用户参考[IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/122271)

## 环境依赖

本目录下的代码均在4卡Tesla K40/P40 GPU，CentOS系统，CUDA-9.0/8.0，cuDNN-7.0环境下测试运行无误

  - Python == 2.7 / 3.6
  - PaddlePaddle >= 1.1.0
  - opencv-python >= 3.3

### 说明

目前已发现在PaddlePaddle 1.3.0 / cuDNN-7.0环境下，存在问题会导致模型训练loss不收敛。推荐使用最新版本PaddlePaddle (>= 1.4).

## MPII Val结果
| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1| Models |
| ---- |:----:|:--------:|:-----:|:-----:|:---:|:----:|:-----:|:----:|:-------:|:------:|
| 256x256\_pose\_resnet\_50 in PyTorch | 96.351	| 95.329 | 88.989 | 83.176 | 88.420	| 83.960 | 79.594 | 88.532 | 33.911 | - |
| 256x256\_pose\_resnet\_50 in Fluid   | 96.385 | 95.363 | 89.211 | 84.084 | 88.454 | 84.182 | 79.546 | 88.748 | 33.750 | [`link`](https://paddlemodels.bj.bcebos.com/pose/pose-resnet50-mpii-256x256.tar.gz) |
| 384x384\_pose\_resnet\_50 in PyTorch | 96.658 | 95.754 | 89.790 | 84.614 | 88.523 | 84.666 | 79.287 | 89.066 | 38.046 | - |
| 384x384\_pose\_resnet\_50 in Fluid   | 96.862 | 95.635 | 90.046 | 85.557 | 88.818 | 84.948 | 78.484 | 89.235 | 38.093 | [`link`](https://paddlemodels.bj.bcebos.com/pose/pose-resnet50-mpii-384x384.tar.gz) |

## COCO val2017结果（使用的检测器在COCO val2017数据集上AP为56.4）
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) | Models |
| ---- |:--:|:-----:|:------:|:------:|:------:|:--:|:-----:|:------:|:------:|:------:|:------:|
| 256x192\_pose\_resnet\_50 in PyTorch | 0.704 | 0.886 | 0.783 | 0.671 | 0.772 | 0.763 | 0.929 | 0.834 | 0.721 | 0.824 | - |
| 256x192\_pose\_resnet\_50 in Fluid   | 0.712 | 0.897 | 0.786 | 0.683 | 0.756 | 0.741 | 0.906 | 0.806 | 0.709 | 0.790 | [`link`](https://paddlemodels.bj.bcebos.com/pose/pose-resnet50-coco-256x192.tar.gz) |
| 384x288\_pose\_resnet\_50 in PyTorch | 0.722 | 0.893 | 0.789 | 0.681 | 0.797 | 0.776 | 0.932 | 0.838 | 0.728 | 0.846 | - |
| 384x288\_pose\_resnet\_50 in Fluid   | 0.727 | 0.897 | 0.796 | 0.690 | 0.783 | 0.754 | 0.907 | 0.813 | 0.714 | 0.814 | [`link`](https://paddlemodels.bj.bcebos.com/pose/pose-resnet50-coco-384x288.tar.gz) |

### 说明

 - 使用Flip test
 - 对当前模型结果并没有进行调参选择，使用下面相关实验配置训练后，取最后一个epoch后的模型作为最终模型，即可得到上述实验结果

## 开始

### 数据准备和预训练模型

 - 安照[提示](https://github.com/Microsoft/human-pose-estimation.pytorch#data-preparation)进行数据准备
 - 下载预训练好的ResNet-50

```bash
wget http://paddle-imagenet-models.bj.bcebos.com/resnet_50_model.tar
```

下载完成后，将模型解压、放入到根目录下的'pretrained'文件夹中，默认文件路径树为：

```
${根目录}
  `-- pretrained
      `-- resnet_50
          |-- 115
  `-- data
      `-- coco
          |-- annotations
          |-- images
      `-- mpii
          |-- annot
          |-- images
```

### 安装 [COCOAPI](https://github.com/cocodataset/cocoapi)

```bash
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
```

### 模型验证（COCO或MPII）

下载COCO/MPII预训练模型（见上表最后一列所附链接），保存到根目录下的'checkpoints'文件夹中，运行：

```bash
python val.py --dataset 'mpii' --checkpoint 'checkpoints/pose-resnet50-mpii-384x384' --data_root 'data/mpii'
```

### 模型训练

```bash
python train.py --dataset 'mpii'
```

**说明** 详细参数配置已保存到`lib/mpii_reader.py` 和 `lib/coco_reader.py`文件中，通过设置dataset来选择使用具体的参数配置

### 模型测试（任意图片，使用上述COCO或MPII预训练好的模型）

同时，我们支持使用预训练好的关键点检测模型预测任意图片

将测试图片放入根目录下的'test'文件夹中，执行

```bash
# 默认是MPII数据集
python test.py --checkpoint 'checkpoints/pose-resnet-50-384x384-mpii'
```

`python test.py --help`获取更多的用法。

## 引用

- Simple Baselines for Human Pose Estimation and Tracking in PyTorch [`code`](https://github.com/Microsoft/human-pose-estimation.pytorch#data-preparation)
