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

__all__ = ['MaskAssigner']


class MaskAssigner(object):
    """
    TODO(qingqing): add comments
    Args:
        cfg (AttrDict): All configuration.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def get_mask_rois_and_targets(self, input_rois, sampled_label, feed_vars):
        """
        TODO(dangqingqing): add comments
        Args:
            input_rois (Variable): Input RoI bboxes with shape [P, 4]. P is
                the number of RoIs.
            sampled_label (Variable): .
            feed_vars (dict): The input dictionary consist of ground truth.

        Returns:
            TODO(qingiqng): add comments.
        """

        def check(name):
            if name not in feed_vars or not feed_vars[name]:
                raise ValueError("{} has no {}".format(feed_vars, name))

        check('gt_label')
        check('is_crowd')
        check('gt_mask')
        check('im_info')

        outs = fluid.layers.generate_mask_labels(
            rois=input_rois,
            gt_classes=feed_vars['gt_label'],
            is_crowd=feed_vars['is_crowd'],
            gt_segms=feed_vars['gt_mask'],
            im_info=feed_vars['im_info'],
            labels_int32=sampled_label,
            num_classes=self.cfg.DATA.CLASS_NUM,
            resolution=self.cfg.MASK_HEAD.RESOLUTION)
        # mask_rois = outs[0]
        # roi_has_mask_int32 = outs[1]
        # mask_int32 = outs[2]
        return outs
