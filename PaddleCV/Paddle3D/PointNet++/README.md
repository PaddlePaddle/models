# PointNet++ 分类和语义分割模型

---
## 内容

- [简介](#简介)
- [快速开始](#快速开始)
- [参考文献](#参考文献)
- [版本更新](#版本更新)

## 简介

[PointNet++](https://arxiv.org/abs/1706.02413) 是 Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas 等人提出的，针对3D数据进行分类和语义分割的模型。该模型基于PointNet进行了拓展, 使用分层点集特征学习来提取点云数据的特征，首先通过对输入point进行分组和采样提取局部区域模式，然后使用多层感知器来获取点特征。PointNet++ 还将点特征传播用于语义分割模型，采用基于距离插值和跨级跳转连接的分层传播策略，对点特征进行向上采样，获得所有原始点的点特征。


网络结构如下所示：

<p align="center">
<img src="image/pointnet2.jpg" height=300 width=800 hspace='10'/> <br />
用于点集分类和分割的 PointNet++ 网络结构
</p>

集合抽象层是网络的基本模块，每个集合抽象层由三个关键层构成:采样层、分组层和特征提取层。

- **采样层**：采样层使用最远点采样(FPS)的方法，从输入点中选择一组点，它定义了局部区域的中心。与随机抽样的方法相比，在质心数目相同的情况下，FPS可以更好的覆盖整个点集。

- **分组层**：分组层通过寻找中心体周围的“邻近”点来构造局部区域集。在度量空间采样的点集中，点的邻域由度量距离定义。这种方法被称为“query ball”，它使得局部区域的特征在空间上更加一般化。

- **特征提取层**: 特征提取层使用 mini-PointNet 对分组层给出的各个区域进行特征提取，获得局部特征。



**注意:** PointNet++ 模型构建依赖于自定义的 C++ 算子，目前仅支持GPU设备在Linux/Unix系统上进行编译，本模型**不能运行在Windows系统或CPU设备上**


## 快速开始

### 安装

**安装 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle):**

在当前目录下运行样例代码需要 PaddelPaddle Fluid v.1.6 或以上的版本. 如果你的运行环境中的 PaddlePaddle 低于此版本, 请根据[安装文档](http://www.paddlepaddle.org/documentation/docs/en/1.6/beginners_guide/install/index_en.html) 中的说明来更新 PaddlePaddle.

### 数据准备

**ModelNet40 数据集:**

PointNet++ 分类模型在 [ModelNet40 数据集](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)上进行训练，我们提供了数据集下载脚本：

```
cd dataset/ModelNet40
sh download.sh
```

数据目录结构如下所示：

```
  dataset/ModelNet40/modelnet40_ply_hdf5_2048
  ├── train_files.txt
  ├── test_files.txt
  ├── shape_names.txt
  ├── ply_data_train0.h5
  ├── ply_data_train_0_id2file.json
  ├── ply_data_test0.h5
  ├── ply_data_test_0_id2file.json
  |   ...

```

**Indoor3DSemSeg 数据集:**

PointNet++ 分类模型在 [Indoor3DSemSeg 数据集](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip)上进行训练，我们提供了数据集下载脚本：

```
cd dataset/Indoor3DSemSeg
sh download.sh
```

数据目录结构如下所示：

```
  dataset/Indoor3DSemSeg/
  ├── all_files.txt
  ├── room_filelist.txt
  ├── ply_data_all_0.h5
  ├── ply_data_all_1.h5
  |   ...

```

### 编译自定义算子

PaddlePaddle Fluid 从 v1.6 版本开始支持自定义算子实现，请确保你的 Paddle 版本不低于 v1.6

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

**分类模型:**

可通过如下方式启动 PointNet++ 分类模型的训练：

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
python train_cls.py --model=MSG --batch_size=16 --save_dir=checkpoints_msg_cls
```

我们同时提供了训练分类模型的“快速开始”脚本：

```
sh scripts/train_cls.sh
```

**语义分割模型:**

可通过如下方式启动 PointNet++ 语义分割模型的训练：

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
python train_seg.py --model=MSG --batch_size=32 --save_dir=checkpoints_msg_seg
```

我们同时提供了训练语义分割模型的“快速开始”脚本：

```
sh scripts/train_seg.sh
```

### 模型评估

**分类模型:**

可通过如下方式启动 PointNet++ 分类模型的评估：

```
# 指定单卡GPU
export CUDA_VISIBLE_DEVICES=0

# 设置动态库的路径到 LD_LIBRARY_PATH 中
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`

# 对给定权重进行评估
python eval_cls.py --model=MSG --weights=checkpoints_cls/200
```

我们同时提供了评估分类模型的“快速开始”脚本：

```
sh scripts/eval_cls.sh
```

分类模型的评估结果如下所示：

| model | Top-1 | download |
| :----- | :---: | :---: |
| SSG(Single-Scale Group) | 87.6 | [model]() |
| MSG(Multi-Scale Group)  | 89.2 | [model]() |

**语义分割模型:**

可通过如下方式启动 PointNet++ 语义分割模型的评估：

```
# 指定单卡GPU
export CUDA_VISIBLE_DEVICES=0

# 设置动态库的路径到 LD_LIBRARY_PATH 中
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`

# 对给定权重进行评估
python eval_seg.py --model=MSG --weights=checkpoints_seg/200
```

我们同时提供了评估语义分割模型的“快速开始”脚本：

```
sh scripts/eval_seg.sh
```

语义分割模型的评估结果如下所示：

| model | Top-1 | download |
| :----- | :---: | :---: |
| SSG(Single-Scale Group) | 86.1 | [model]() |
| MSG(Multi-Scale Group)  | 86.8 | [model]() |

## 参考文献

- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413), Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas.
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://www.semanticscholar.org/paper/PointNet%3A-Deep-Learning-on-Point-Sets-for-3D-and-Qi-Su/d997beefc0922d97202789d2ac307c55c2c52fba), Charles Ruizhongtai Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas.

## 版本更新

- 11/2019, 新增 PointNet++ 分类和语义分割模型。
