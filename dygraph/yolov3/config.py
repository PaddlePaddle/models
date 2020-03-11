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

# Snapshot period
_C.snapshot_iter = 2000

# min valid area for gt boxes
_C.gt_min_area = -1

# max target box number in an image
_C.max_box_num = 50

#
# Training options
#

# valid score threshold to include boxes
_C.valid_thresh = 0.005

# threshold vale for box non-max suppression
_C.nms_thresh = 0.45

# the number of top k boxes to perform nms
_C.nms_topk = 400

# the number of output boxes after nms
_C.nms_posk = 100

# score threshold for draw box in debug mode
_C.draw_thresh = 0.5

#
# Model options
#

# pixel mean values
_C.pixel_means = [0.485, 0.456, 0.406]

# pixel std values
_C.pixel_stds = [0.229, 0.224, 0.225]

# anchors box weight and height
_C.anchors = [
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326
]

# anchor mask of each yolo layer
_C.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# IoU threshold to ignore objectness loss of pred box
_C.ignore_thresh = .7

#
# SOLVER options
#

# batch size
_C.batch_size = 8

# derived learning rate the to get the final learning rate.
_C.learning_rate = 0.001

# maximum number of iterations
_C.max_iter = 500200

# warm up to learning rate 
_C.warm_up_iter = 4000
_C.warm_up_factor = 0.

# lr steps_with_decay
_C.lr_steps = [400000, 450000]
_C.lr_gamma = 0.1

# L2 regularization hyperparameter
_C.weight_decay = 0.0005

# momentum with SGD
_C.momentum = 0.9

#
# ENV options
#

# support both CPU and GPU
_C.use_gpu = True

# Class number
_C.class_num = 80

# dataset path
_C.train_file_list = 'annotations/instances_train2017.json'
_C.train_data_dir = 'train2017'
_C.val_file_list = 'annotations/instances_val2017.json'
_C.val_data_dir = 'val2017'


def merge_cfg_from_args(args):
    """Merge config keys, values in args into the global config."""
    for k, v in sorted(six.iteritems(vars(args))):
        try:
            value = eval(v)
        except:
            value = v
        _C[k] = value
