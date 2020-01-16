# -*- coding: utf-8 -*-
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from __future__ import unicode_literals
from .collect import SegConfig
import numpy as np

cfg = SegConfig()

########################## 基本配置 ###########################################
# 均值，图像预处理减去的均值
#cfg.MEAN = [0.5, 0.5, 0.5]
cfg.MEAN = [0.485, 0.456, 0.406]
# 标准差，图像预处理除以标准差·
cfg.STD = [0.229, 0.224, 0.225]
# 批处理大小
cfg.TRAIN_BATCH_SIZE_PER_GPU = 2
cfg.TRAIN_BATCH_SIZE= 8
cfg.EVAL_BATCH_SIZE= 8
# 多进程训练总进程数
cfg.NUM_TRAINERS = 1
# 多进程训练进程ID
cfg.TRAINER_ID = 0
########################## 数据载入配置 #######################################
# 数据载入时的并发数, 建议值8
cfg.DATALOADER.NUM_WORKERS = 8
# 数据载入时缓存队列大小, 建议值256
cfg.DATALOADER.BUF_SIZE = 256

########################## 数据集配置 #########################################
cfg.DATASET.DATASET_NAME = 'cityscapes'
# 数据主目录目录
cfg.DATASET.DATA_DIR = './data_local/cityscapes/'
# 训练集列表
cfg.DATASET.TRAIN_FILE_LIST = './data_local/cityscapes/train.list'
# 训练集数量
cfg.DATASET.TRAIN_TOTAL_IMAGES = 5
# 验证集列表
cfg.DATASET.VAL_FILE_LIST = './data_local/cityscapes/val.list'
# 验证数据数量
cfg.DATASET.VAL_TOTAL_IMAGES = 50
# 测试数据列表
cfg.DATASET.TEST_FILE_LIST = './data_local/cityscapes/test.list'
# 测试数据数量
cfg.DATASET.TEST_TOTAL_IMAGES = 1525
# Tensorboard 可视化的数据集
cfg.DATASET.VIS_FILE_LIST = None
# 类别数(需包括背景类)
cfg.DATASET.NUM_CLASSES = 19
# 输入图像类型, 支持三通道'rgb',四通道'rgba',单通道灰度图'gray'
cfg.DATASET.IMAGE_TYPE = 'rgb'
# 输入图片的通道数
cfg.DATASET.DATA_DIM = 3
# 数据列表分割符, 默认为空格
cfg.DATASET.SEPARATOR = '\t'
# 忽略的像素标签值, 默认为255，一般无需改动
cfg.DATASET.IGNORE_INDEX = 255
# 数据增强是图像的padding值
cfg.DATASET.PADDING_VALUE = [127.5, 127.5, 127.5]

########################### 数据增强配置 ######################################
cfg.DATAAUG.EXTRA = True
cfg.DATAAUG.BASE_SIZE = 1024
cfg.DATAAUG.CROP_SIZE = 769
cfg.DATAAUG.RAND_SCALE_MIN = 0.75
cfg.DATAAUG.RAND_SCALE_MAX = 2.0


########################### 训练配置 ##########################################
# 模型保存路径
cfg.TRAIN.MODEL_SAVE_DIR = ''
# 预训练模型路径
cfg.TRAIN.PRETRAINED_MODEL_DIR = ''
# 是否resume，继续训练
cfg.TRAIN.RESUME_MODEL_DIR = ''
# 是否使用多卡间同步BatchNorm均值和方差
cfg.TRAIN.SYNC_BATCH_NORM = True
# 模型参数保存的epoch间隔数，可用来继续训练中断的模型
cfg.TRAIN.SNAPSHOT_EPOCH = 10

########################### 模型优化相关配置 ##################################
# 初始学习率
cfg.SOLVER.LR = 0.001
# 学习率下降方法, 支持poly piecewise cosine 三种
cfg.SOLVER.LR_POLICY = "poly"
# 优化算法, 支持SGD和Adam两种算法
cfg.SOLVER.OPTIMIZER = "sgd"
# 动量参数
cfg.SOLVER.MOMENTUM = 0.9
# 二阶矩估计的指数衰减率
cfg.SOLVER.MOMENTUM2 = 0.999
# 学习率Poly下降指数
cfg.SOLVER.POWER = 0.9
# step下降指数
cfg.SOLVER.GAMMA = 0.1
# step下降间隔
cfg.SOLVER.DECAY_EPOCH = [10, 20]
# 学习率权重衰减，0-1
#cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.WEIGHT_DECAY = 0.00004
# 训练开始epoch数，默认为1
cfg.SOLVER.BEGIN_EPOCH = 1
# 训练epoch数，正整数
cfg.SOLVER.NUM_EPOCHS = 30
# loss的选择，支持softmax_loss, bce_loss, dice_loss
cfg.SOLVER.LOSS = ["softmax_loss"]
# 是否开启warmup学习策略 
cfg.SOLVER.LR_WARMUP = False 
# warmup的迭代次数
cfg.SOLVER.LR_WARMUP_STEPS = 2000 

########################## 测试配置 ###########################################
# 测试模型路径
cfg.TEST.TEST_MODEL = ''
cfg.TEST.BASE_SIZE = 2048
cfg.TEST.CROP_SIZE = 769
cfg.TEST.SLIDE_WINDOW = True

########################## 模型通用配置 #######################################
# 模型名称, 支持pspnet, deeplabv3, glore, ginet 
cfg.MODEL.MODEL_NAME = ''
# BatchNorm类型: bn、gn(group_norm)
cfg.MODEL.DEFAULT_NORM_TYPE = 'bn'
# 多路损失加权值
cfg.MODEL.MULTI_LOSS_WEIGHT = [1.0, 0.4]
# DEFAULT_NORM_TYPE为gn时group数
cfg.MODEL.DEFAULT_GROUP_NUMBER = 32
# 极小值, 防止分母除0溢出，一般无需改动
cfg.MODEL.DEFAULT_EPSILON = 1e-5
# BatchNorm动量, 一般无需改动
cfg.MODEL.BN_MOMENTUM = 0.99
# 是否使用FP16训练
cfg.MODEL.FP16 = False
# 混合精度训练需对LOSS进行scale, 默认为动态scale，静态scale可以设置为512.0
cfg.MODEL.SCALE_LOSS = "DYNAMIC"
# backbone network, (resnet, hrnet, xception_65, mobilenetv2)
cfg.MODEL.BACKBONE= "resnet"
#  backbone_layer: 101 and 50 for resnet
cfg.MODEL.BACKBONE_LAYERS=101
# strides= input.size / feature_maps.size
cfg.MODEL.BACKBONE_OUTPUT_STRIDE=8
cfg.MODEL.BACKBONE_MULTI_GRID = False



########################## PSPNET模型配置 ######################################
# RESNET backbone scale 设置
cfg.MODEL.PSPNET.DEPTH_MULTIPLIER = 1
# Aux loss 
cfg.MODEL.PSPNET.AuxHead= True


########################## GloRe模型配置 ######################################
# RESNET backbone scale 设置
cfg.MODEL.GLORE.DEPTH_MULTIPLIER = 1
# Aux loss 
cfg.MODEL.GLORE.AuxHead= True

########################## DeepLabv3模型配置 ####################################
# MobileNet v2 backbone scale 设置
cfg.MODEL.DEEPLABv3.DEPTH_MULTIPLIER = 1.0
# ASPP是否使用可分离卷积
cfg.MODEL.DEEPLABv3.ASPP_WITH_SEP_CONV = True
cfg.MODEL.DEEPLABv3.AuxHead= True



########################## HRNET模型配置 ######################################
# HRNET STAGE2 设置
cfg.MODEL.HRNET.STAGE2.NUM_MODULES = 1
cfg.MODEL.HRNET.STAGE2.NUM_CHANNELS = [40, 80]
# HRNET STAGE3 设置
cfg.MODEL.HRNET.STAGE3.NUM_MODULES = 4
cfg.MODEL.HRNET.STAGE3.NUM_CHANNELS = [40, 80, 160]
# HRNET STAGE4 设置
cfg.MODEL.HRNET.STAGE4.NUM_MODULES = 3
cfg.MODEL.HRNET.STAGE4.NUM_CHANNELS = [40, 80, 160, 320]


