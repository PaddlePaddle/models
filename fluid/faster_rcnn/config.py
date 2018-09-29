#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import six


class EnvConfig(object):
    # support both CPU and GPU
    use_gpu = True
    # Whether use parallel
    parallel = True
    # Class number
    class_num = 81
    # support pyreader
    use_pyreader = True
    # pixel mean values
    pixel_means = [102.9801, 115.9465, 122.7717]
    # clip box to prevent overflowing
    bbox_clip = np.log(1000. / 16.)


class SolverConfig(object):
    # derived learning rate the to get the final learning rate.
    learning_rate = 0.01
    # maximum number of iterations
    max_iter = 180000
    # warm up to learning rate 
    warm_up_iter = 500
    warm_up_factor = 1. / 3.
    # lr steps_with_decay
    lr_steps = [120000, 160000]
    lr_gamma = 0.1
    # L2 regularization hyperparameter
    weight_decay = 0.0001
    # momentum with SGD
    momentum = 0.9


class TrainConfig(object):
    # scales an image's shortest side
    scales = [800]
    # max size of longest side
    max_size = 1333
    # images per GPU in minibatch
    im_per_batch = 1
    # roi minibatch size per image
    batch_size_per_im = 512
    # target fraction of foreground roi minibatch 
    fg_fractrion = 0.25
    # overlap threshold for a foreground roi
    fg_thresh = 0.5
    # overlap threshold for a background roi
    bg_thresh_hi = 0.5
    bg_thresh_lo = 0.0
    # If False, only resize image and not pad, image shape is different between
    # GPUs in one mini-batch. If True, image shape is the same in one mini-batch.
    padding_minibatch = False
    # Snapshot period
    snapshot_iter = 10000
    # number of RPN proposals to keep before NMS
    rpn_pre_nms_top_n = 12000
    # number of RPN proposals to keep after NMS
    rpn_post_nms_top_n = 2000
    # NMS threshold used on RPN proposals
    rpn_nms_thresh = 0.7
    # min size in RPN proposals
    rpn_min_size = 0.0
    # eta for adaptive NMS in RPN
    rpn_eta = 1.0
    # number of RPN examples per image
    rpn_batch_size_per_im = 256
    # remove anchors out of the image
    rpn_straddle_thresh = 0.
    # target fraction of foreground examples pre RPN minibatch
    rpn_fg_fraction = 0.5
    # min overlap between anchor and gt box to be a positive examples
    rpn_positive_overlap = 0.7
    # max overlap between anchor and gt box to be a negative examples
    rpn_negative_overlap = 0.3
    # stopgrad at a specified stage
    freeze_at = 2
    # min area of ground truth box
    gt_min_area = -1


class InferConfig(object):
    # scales an image's shortest side
    scales = [800]
    # max size of longest side
    max_size = 1333
    # eta for adaptive NMS in RPN
    rpn_eta = 1.0
    # min score threshold to infer
    score_thresh = 0.05
    # overlap threshold used for NMS
    nms_thresh = 0.5
    # number of RPN proposals to keep before NMS
    rpn_pre_nms_top_n = 6000
    # number of RPN proposals to keep after NMS
    rpn_post_nms_top_n = 1000
    # min size in RPN proposals
    rpn_min_size = 0.0
    # max number of detections
    detectiions_per_im = 100
    # NMS threshold used on RPN proposals
    rpn_nms_thresh = 0.7


class ModelConfig(object):
    # weight for bbox regression targets
    bbox_reg_weights = [0.1, 0.1, 0.2, 0.2]
    # RPN anchor sizes
    anchor_sizes = [32, 64, 128, 256, 512]
    # RPN anchor ratio
    aspect_ratio = [0.5, 1, 2]
    # variance of anchors
    variances = [1., 1., 1., 1.]
    # stride of feature map
    rpn_stride = [16.0, 16.0]


def merge_cfg_from_list(args, g_cfgs):
    """
    Set the above global configurations using the cfg_list. 
    """
    for key, value in sorted(six.iteritems(vars(args))):
        for g_cfg in g_cfgs:
            if hasattr(g_cfg, key):
                try:
                    value = eval(value)
                except Exception:  # for file path
                    pass
                setattr(g_cfg, key, value)
                break
