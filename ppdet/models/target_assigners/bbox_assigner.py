#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.fluid as fluid

__all__ = ['BBoxAssigner']


class BBoxAssigner(object):
    def __init__(self, cfg):
        """
        Args:
            cfg(dict): All parameters in dictionary
        """
        self.cfg = cfg
        self.use_random = getattr(cfg.TRAIN, 'RANDOM', False)
        self.class_num = cfg.DATA.CLASS_NUM

        local_cfg = cfg.RPN_HEAD.PROPOSAL

        self.batch_size_per_im = local_cfg.BATCH_SIZE_PER_IM
        self.fg_fractrion = local_cfg.FG_FRACTION
        self.fg_thresh = local_cfg.FG_THRESH
        self.bg_thresh_hi = local_cfg.BG_THRESH_HI
        self.bg_thresh_lo = local_cfg.BG_THRESH_LO
        self.bbox_reg_weights = local_cfg.BBOX_REG_WEIGHTS

    def get_sampled_rois_and_targets(self, input_rois, feed_vars):
        outs = fluid.layers.generate_proposal_labels(
            rpn_rois=input_rois,
            gt_classes=feed_vars['gt_label'],
            is_crowd=feed_vars['is_crowd'],
            gt_boxes=feed_vars['gt_box'],
            im_info=feed_vars['im_info'],
            batch_size_per_im=self.batch_size_per_im,
            fg_fraction=self.fg_fractrion,
            fg_thresh=self.fg_thresh,
            bg_thresh_hi=self.bg_thresh_hi,
            bg_thresh_lo=self.bg_thresh_lo,
            bbox_reg_weights=self.bbox_reg_weights,
            class_nums=self.class_num,
            use_random=self.use_random)
        # rois = outs[0]
        # labels_int32 = outs[1]
        # bbox_targets = outs[2]
        # bbox_inside_weights = outs[3]
        # bbox_outside_weights = outs[4]
        return outs
