#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License. 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from edict import AttrDict
import six
import numpy as np

_C = AttrDict()
cfg = _C

#
# Training options
#
_C.TRAIN = AttrDict()

# scales an image's shortest side
_C.TRAIN.scales = [800]

# max size of longest side
_C.TRAIN.max_size = 1333

# images per GPU in minibatch
_C.TRAIN.im_per_batch = 1

# roi minibatch size per image
_C.TRAIN.batch_size_per_im = 256

# target fraction of foreground roi minibatch 
_C.TRAIN.fg_fractrion = 0.25

# overlap threshold for a foreground roi
_C.TRAIN.fg_thresh = 0.5

# overlap threshold for a background roi
_C.TRAIN.bg_thresh_hi = 0.5
_C.TRAIN.bg_thresh_lo = 0.0

# If False, only resize image and not pad, image shape is different between
# GPUs in one mini-batch. If True, image shape is the same in one mini-batch.
_C.TRAIN.padding_minibatch = False

# Snapshot period
_C.TRAIN.snapshot_iter = 1000

# number of RPN proposals to keep before NMS
_C.TRAIN.rpn_pre_nms_top_n = 12000

# number of RPN proposals to keep after NMS
_C.TRAIN.rpn_post_nms_top_n = 2000

# NMS threshold used on RPN proposals
_C.TRAIN.rpn_nms_thresh = 0.7

# min size in RPN proposals
_C.TRAIN.rpn_min_size = 0.0

# eta for adaptive NMS in RPN
_C.TRAIN.rpn_eta = 1.0

# number of RPN examples per image
_C.TRAIN.rpn_batch_size_per_im = 256

# remove anchors out of the image
_C.TRAIN.rpn_straddle_thresh = 0.

# target fraction of foreground examples pre RPN minibatch
_C.TRAIN.rpn_fg_fraction = 0.5

# min overlap between anchor and gt box to be a positive examples
_C.TRAIN.rpn_positive_overlap = 0.7

# max overlap between anchor and gt box to be a negative examples
_C.TRAIN.rpn_negative_overlap = 0.3

# stopgrad at a specified stage
_C.TRAIN.freeze_at = 2

# min area of ground truth box
_C.TRAIN.gt_min_area = -1

#
# Inference options
#
_C.TEST = AttrDict()

# scales an image's shortest side
_C.TEST.scales = [800]

# max size of longest side
_C.TEST.max_size = 1333

# eta for adaptive NMS in RPN
_C.TEST.rpn_eta = 1.0

# min score threshold to infer
_C.TEST.score_thresh = 0.01

# overlap threshold used for NMS
_C.TEST.nms_thresh = 0.3

# number of RPN proposals to keep before NMS
_C.TEST.rpn_pre_nms_top_n = 6000

# number of RPN proposals to keep after NMS
_C.TEST.rpn_post_nms_top_n = 1000

# min size in RPN proposals
_C.TEST.rpn_min_size = 0.0

# max number of detections
_C.TEST.detections_per_im = 300

# NMS threshold used on RPN proposals
_C.TEST.rpn_nms_thresh = 0.7

#
# Model options
#

# Whether use mask rcnn head
_C.MASK_ON = True

# weight for bbox regression targets
_C.bbox_reg_weights = [10.0, 10.0, 5.0, 5.0, 1.0]

# RPN anchor sizes
_C.anchor_sizes = [128, 256, 512]

# RPN anchor ratio
_C.aspect_ratio = [0.2, 0.5, 1.0]

# RPN anchor angle
_C.anchor_angle = [-30.0, 0.0, 30.0, 60.0, 90.0, 120.0]

# variance of anchors
_C.variances = [1., 1., 1., 1., 1.]

# stride of feature map
_C.rpn_stride = [16.0, 16.0]

# pooled width and pooled height 
_C.roi_resolution = 14

# spatial scale 
_C.spatial_scale = 1. / 16.

# resolution to represent rotated roi align
_C.resolution = 14

#
# SOLVER options
#

# derived learning rate the to get the final learning rate.
_C.learning_rate = 0.01

# maximum number of iterations
_C.max_iter = 140000

# warm up to learning rate 
_C.warm_up_iter = 500
_C.start_factor = 1. / 3

# lr steps_with_decay
_C.lr_steps = [6250, 12500]
_C.lr_gamma = 0.1

# L2 regularization hyperparameter
_C.weight_decay = 0.0001

# momentum with SGD
_C.momentum = 0.9

#
# ENV options
#

# support both CPU and GPU
_C.use_gpu = True

# Whether use parallel
_C.parallel = True

# Class number
_C.class_num = 81

# support pyreader
_C.use_pyreader = True
_C.TRAIN.min_size = 800
_C.TRAIN.max_size = 1333
_C.TEST.min_size = 1000
# pixel mean values
_C.pixel_means = [0.485, 0.456, 0.406]
_C.pixel_std = [0.229, 0.224, 0.225]
# clip box to prevent overflowing
_C.bbox_clip = np.log(1000. / 16.)


def merge_cfg_from_args(args, mode):
    """Merge config keys, values in args into the global config."""
    if mode == 'train':
        sub_d = _C.TRAIN
    else:
        sub_d = _C.TEST
    for k, v in sorted(six.iteritems(vars(args))):
        d = _C
        try:
            value = eval(v)
        except:
            value = v
        if k in sub_d:
            sub_d[k] = value
        else:
            d[k] = value
