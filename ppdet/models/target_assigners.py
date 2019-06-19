# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid

from ppdet.core.workspace import register

__all__ = ['BBoxAssigner', 'MaskAssigner']


@register
class BBoxAssigner(object):
    __op__ = fluid.layers.generate_proposal_labels
    __append_doc__ = True

    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=.5,
                 bg_thresh_hi=.5,
                 bg_thresh_lo=0.,
                 bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                 num_classes=81,
                 shuffle_before_sample=True):
        super(BBoxAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        self.bbox_reg_weights = bbox_reg_weights
        self.class_nums = num_classes
        self.use_random = shuffle_before_sample


@register
class MaskAssigner(object):
    __op__ = fluid.layers.generate_mask_labels
    __append_doc__ = True

    def __init__(self, num_classes=81, resolution=14):
        super(MaskAssigner, self).__init__()
        self.num_classes = num_classes
        self.resolution = resolution
