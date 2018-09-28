# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
#
# Based on:
# --------------------------------------------------------
# Detectron
# Copyright (c) 2017-present, Facebook, Inc.
# Licensed under the Apache License, Version 2.0;
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future.utils import iteritems
from past.builtins import basestring
from ast import literal_eval
import numpy as np
import os
import os.path as osp
import yaml
from collect import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from detectron.core.config import cfg
cfg = __C

# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Scales to use during training
# Each scale is the pixel size of an image's shortest side
# If multiple scales are listed, then one is selected uniformly at random for
# each training image (i.e., scale jitter data augmentation)
__C.TRAIN.SCALES = [800]

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1333

# Images *per GPU* in the training minibatch
# Total images per minibatch = TRAIN.IMS_PER_BATCH * NUM_GPUS
__C.TRAIN.IMS_PER_BATCH = 1

# RoI minibatch size *per image* (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE_PER_IM = 512

# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for an RoI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for an RoI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Snapshot (model checkpoint) period
# Divide by NUM_GPUS to determine actual period (e.g., 20000/8 => 2500 iters)
# to allow for linear training schedule scaling
__C.TRAIN.SNAPSHOT_ITERS = 10000

# If False, only resize image and not pad, image shape is different between
# GPUs in one mini-batch. If True, image shape is the same in one mini-batch.
__C.TRAIN.PADDING_MINIBATCH = False

# ---------------------------------------------------------------------------- #
# RPN training options
# ---------------------------------------------------------------------------- #

# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000

# Number of top scoring RPN proposals to keep after applying NMS
# This is the total number of RPN proposals produced (for both FPN and non-FPN
# cases)
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

# NMS threshold used on RPN proposals (used during end-to-end training with RPN)
__C.TRAIN.RPN_NMS_THRESH = 0.7

# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (at orig image scale; not scale used during training or inference)
__C.TRAIN.RPN_MIN_SIZE = 0.0

# eta for adaptive nms in RPN
__C.TRAIN.RPN_ETA = 1.0

# Total number of RPN examples per image
__C.TRAIN.RPN_BATCH_SIZE_PER_IM = 256

# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
__C.TRAIN.RPN_STRADDLE_THRESH = 0.

# Target fraction of foreground (positive) examples per RPN minibatch
__C.TRAIN.RPN_FG_FRACTION = 0.5

# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IOU >= thresh ==> positive RPN
# example)
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IOU < thresh ==> negative RPN
# example)
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Add StopGrad at a specified stage so the bottom layers are frozen
__C.TRAIN.FREEZE_AT = 2

# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

# Scale to use during testing
__C.TEST.SCALES = 800

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1333

# eta for adaptive nms in RPN
__C.TEST.RPN_ETA = 1.0

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
__C.TEST.SCORE_THRESH = 0.05

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.5

# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

# Number of top scoring RPN proposals to keep after applying NMS
# This is the total number of RPN proposals produced (for both FPN and non-FPN
# cases)
__C.TEST.RPN_POST_NMS_TOP_N = 1000

# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (at orig image scale; not scale used during training or inference)
__C.TEST.RPN_MIN_SIZE = 0.

# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
__C.TEST.DETECTIONS_PER_IM = 100

# NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# Number of classes in the dataset; must be set
# E.g., 81 for COCO (80 foreground + 1 background)
__C.MODEL.NUM_CLASSES = 81

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
__C.MODEL.BBOX_REG_WEIGHTS = [0.1, 0.1, 0.2, 0.2]

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
__C.RPN = AttrDict()

# RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
# Note: these options are *not* used by FPN RPN; see FPN.RPN* options
__C.RPN.SIZES = [32, 64, 128, 256, 512]

# Stride of the feature map that RPN is attached
__C.RPN.STRIDE = [16.0, 16.0]

# RPN anchor aspect ratios
__C.RPN.ASPECT_RATIOS = [0.5, 1, 2]

#The variance of anchors
__C.RPN.VARIANCES = [1., 1., 1., 1.]

# ---------------------------------------------------------------------------- #
# Solver options
# Note: all solver options are used exactly as specified; the implication is
# that if you switch from training on 1 GPU to N GPUs, you MUST adjust the
# solver configuration accordingly. We suggest using gradual warmup and the
# linear learning rate scaling rule as described in
# "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" Goyal et al.
# https://arxiv.org/abs/1706.02677
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.01

# Maximum number of SGD iterations
__C.SOLVER.MAX_ITER = 180000

# Warm up to SOLVER.BASE_LR over this number of SGD iterations
__C.SOLVER.WARM_UP_ITERS = 500

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARM_UP_FACTOR
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 3.0

# 'steps_with_decay'
__C.SOLVER.STEPS = [120000, 160000]
__C.SOLVER.GAMMA = 0.1

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0001

# Momentum to use with SGD
__C.SOLVER.MOMENTUM = 0.9

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# "Fun" fact: the history of where these values comes from is lost
__C.PIXEL_MEANS = [102.9801, 115.9465, 122.7717]

# Clip bounding box transformation predictions to prevent np.exp from
# overflowing
# Heuristic choice based on that would scale a 16 pixel anchor up to 1000 pixels
__C.BBOX_XFORM_CLIP = np.log(1000. / 16.)
