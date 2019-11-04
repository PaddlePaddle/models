# PointNet++ classification and semantic segmentation model

---
## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [FAQ](#faq)
- [Reference](#reference)
- [Update](#update)

## Introduction

[PointNet++](https://arxiv.org/abs/1706.02413) is a point classification and segmentation model for 3D data proposed by Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas.
This model is a extension work based on PointNet.

**NOTE:** PointNet++ model builds base on custom C++ operations, which can only support GPU devices and compiled on Linux/Unix currently, this model **cannot run on Windows or CPU deivices**.


## Quick Start

### Installation

**Install [PaddlePaddle](https://github.com/PaddlePaddle/Paddle):**

Running sample code in this directory requires PaddelPaddle Fluid v.1.6 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in [installation document](http://www.paddlepaddle.org/documentation/docs/en/1.6/beginners_guide/install/index_en.html) and make an update.

### Data preparation

**ModelNet40 dataset:**

PointNet++ classification models are reproduced on [ModelNet40 dataset](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), we also provide download scripts as follows:

```
cd dataset/ModelNet40
sh download.sh
```

The dataset catalog structure is as follows:

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

**Indoor3DSemSeg dataset:**

PointNet++ semantic segmentation models are reproduced on [Indoor3DSemSeg dataset](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip), we also provide download scripts as follows:

```
cd dataset/Indoor3DSemSeg
sh download.sh
```

The dataset catalog structure is as follows:

```
  dataset/Indoor3DSemSeg/
  ├── all_files.txt
  ├── room_filelist.txt
  ├── ply_data_all_0.h5
  ├── ply_data_all_1.h5
  |   ...

```

### Compile custom operations

Custom operations are supported since Paddle Fluid v1.6, please make sure you are using Paddle not less than v1.6.
Custom operations can be compiled as follows:

```
cd ext_op/src
sh make.sh
```

If the compilation is finished successfully, `pointnet2_lib.so` will be generated under `exr_op/src`.

### Training

**Classification Model:**

For PointNet++ classification model, training can be start as follows:

```
# For single GPU deivces
export CUDA_VISIBLE_DEVICES=0

# enable gc to save GPU memory
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

# export paddle libs to LD_LIBRARY_PATH for custom op library
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`

# start training
python tools/train_cls.py --model=MSG --batch_size=16 --save_dir=checkpoints_msg_cls
```

We also provide quick start script for training classification model as follows:

```
sh scripts/train_cls.sh
```

**Semantic Segmentation Model:**

For PointNet++ semantic segmentation model, training can be start as follows:

```
# For single GPU deivces
export CUDA_VISIBLE_DEVICES=0

# enable gc to save GPU memory
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

# export paddle libs to LD_LIBRARY_PATH for custom op library
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`

# start training
python tools/train_seg.py --model=MSG --batch_size=32 --save_dir=checkpoints_msg_seg
```

We also provide quick start scripts for training semantic segmentation model as follows:

```
sh scripts/train_seg.sh
```

### Evaluation
