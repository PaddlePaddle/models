# PointRCNN 3D目标检测模型

---
## 内容

- [简介](#简介)
- [快速开始](#快速开始)
- [参考文献](#参考文献)
- [版本更新](#版本更新)

## 简介

[PointRCNN](https://arxiv.org/abs/1812.04244) 是 Shaoshuai Shi, Xiaogang Wang, Hongsheng Li. 等人提出的，第一个仅使用原始点云作为输入的两级3D目标检测器，以自下而上的方式直接从原始点云生成精确的定位框的方案，然后基于bin，在标准坐标系中对3D BBox回归损失进行细化。PointRCNN在KITTI数据集上进行评估，并在提交时所有已发布作品中的KITTI 3D目标检测排行榜上获得最佳性能。

网络结构如下所示：

<p align="center">
<img src="doc/teaser.png" height=300 width=800 hspace='10'/> <br />
用于点云的目标检测器 PointNet++
</p>

**注意:** PointRCNN 模型构建依赖于自定义的 C++ 算子，目前仅支持GPU设备在Linux/Unix系统上进行编译，本模型**不能运行在Windows系统或CPU设备上**


## 快速开始

### 安装

**安装 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle):**

在当前目录下运行样例代码需要 PaddelPaddle Fluid v.1.6 或以上的版本. 如果你的运行环境中的 PaddlePaddle 低于此版本, 请根据[安装文档](http://www.paddlepaddle.org/documentation/docs/en/1.6/beginners_guide/install/index_en.html) 中的说明来更新 PaddlePaddle.

为了使自定义算子与paddle版本兼容，建议您**优先使用源码编译paddle**，源码编译方式请参考[编译安装](https://www.paddlepaddle.org.cn/install/doc/source/ubuntu)

### 数据准备

**KITTI 3D object detection 数据集:**

PointRCNN使用数据集[KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) 
上进行训练，

数据目录结构如下所示：

```
PointRCNN
├── data
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2

```

此处的images只用做可视化，可以选择是用road planes(https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing)来做训练时的数据增强
### 编译自定义算子

PaddlePaddle Fluid 从 v1.6 版本开始支持自定义算子实现请，确保你的 Paddle 版本不低于 v1.6。
这里默认您已经使用源码编译安装了paddle，如果您是通过 pip 安装 paddle 版本，请根据 ext_op/README.md 对编译脚本进行修改。

自定义算子可通过以下方式进行编译：

```
cd ext_op/src
sh make.sh
```
成功编译后，`exr_op/src` 目录下将会生成 `pointnet2_lib.so` 

执行下列操作，确保自定义算子编译正确：

```
# 设置动态库的路径到 LD_LIBRARY_PATH 中
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`

# 回到 ext_op 目录，添加 PYTHONPATH
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`

# 运行单测 
python test/test_farthest_point_sampling_op.py
python test/test_gather_point_op.py
python test/test_group_points_op.py
python test/test_query_ball_op.py
python test/test_three_interp_op.py
python test/test_three_nn_op.py
```
单测运行成功会输出提示信息，如下所示：

```
.
----------------------------------------------------------------------
Ran 1 test in 13.205s

OK
```

更多关于自定义算子的编译说明，可阅读 ext_op/README.md


### 训练

**PointRCNN模型:**

可通过如下方式启动 PointRCNN模型的训练：

```
# 指定单卡GPU训练
export CUDA_VISIBLE_DEVICES=0

# 开启 gc 节省显存
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

# 设置动态库的路径到 LD_LIBRARY_PATH 中
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`

# 开始训练

首先要生成Ground Truth采样数据

python generate_gt_database.py --class_name 'Car' --split train
 
# 训练 RPN 阶段

python tools/train.py --cfg=./cfgs/default.yml --train_mode=rpn --batch_size=16 --epoch=200

训练完之后，会默认保存rpn训练的checkpoints到checkpoints目录，也可以通过--save_dir来指定

# 训练 RCNN 阶段

首先，通过一下步骤(1,2)生成rcnn训练需要的roi和feature

1.生成增强的离线场景数据

python tools/generate_aug_scene.py --class_name 'Car' --split train --aug_times 4

2.保存RPN的features 和 roi,可以通过参数--ckpt_dir来指定RPN训练阶段保存权重的路径，默认是tools/train.py中—save_dir指定的路径

python tools/eval.py --cfg=cfgs/default.yaml --batch_size=4 --eval_mode=rpn --ckpt_dir=./checkpoints/ --save_rpn_feature 

# 为train阶段保存 features and roi，在配置文件中设置`TEST.RPN_POST_NMS_TOP_N=300` and `TEST.RPN_NMS_THRESH=0.85`

python tools/eval.py --cfg=cfgs/default.yaml --batch_size=4 --eval_mode=rpn --ckpt_dir=./checkpoints/ --save_rpn_feature

# 为eval阶段保存 features and roi,  在配置文件中设置`TEST.RPN_POST_NMS_TOP_N=100` and `TEST.RPN_NMS_THRESH=0.8`

然后开始训练rcnn，并且通过参数`--rcnn_training_roi_dir` and `--rcnn_training_feature_dir` 来指定roi和feature的目录

python tools/train.py --cfg=./cfgs/default.yml --train_mode=rcnn_offline --batch_size=4 --epoch=30 --rcnn_training_roi_dir= --rcnn_training_feature_dir=

```

**注意**: 
* 最好的模型是通过CPU版本的建议框采样的离线增强策略训练出来的，目前默认仅支持这种方式


### 模型评估

**PointRCNN模型:**

可通过如下方式启动 PointRCNN 模型的评估：

```
# 指定单卡GPU
export CUDA_VISIBLE_DEVICES=0

# 设置动态库的路径到 LD_LIBRARY_PATH 中
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`

# 对给定权重进行评估
python eval.py --cfg=./cfgs/default.yml --eval_mode=rcnn_offline --batch_size=1 --ckpt_dir=./checkpoints/29

# 运行kitt map工具
这个工具只支持python 3.6以上， 需要通过pip安装Numpy, skim age, Numba, fire 

python3 kitti_map.py
```

评估结果如下所示：

```
Car AP@0.70, 0.70, 0.70:
bbox AP:96.91, 89.53, 88.74
bev  AP:90.21, 87.89, 85.51
3d   AP:89.19, 78.85, 77.91
aos  AP:96.90, 89.41, 88.54
```

## 参考文献

- [PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud](https://arxiv.org/abs/1812.04244), Shaoshuai Shi, Xiaogang Wang, Hongsheng Li.

## 版本更新

- 11/2019, 新增 PointRCNN模型。



