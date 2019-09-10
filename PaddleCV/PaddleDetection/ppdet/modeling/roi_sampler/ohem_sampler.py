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

import paddle.fluid as fluid
import numpy as np

from ppdet.core.workspace import register

__all__ = ['OHEMSampler']


@register
class OHEMSampler(object):
    """
    Online Hard Example Mining sampler.
    Args:
    """

    def __init__(self, batch_size_per_im=512):
        super(OHEMSampler, self).__init__()
        self.batch_size_per_im = batch_size_per_im

    def __call__(self, cls_score, bbox_pred, rpn_outs):
        proposals = rpn_outs[0]
        labels_int32 = rpn_outs[1]
        bbox_targets = rpn_outs[2]
        bbox_inside_weights = rpn_outs[3]
        bbox_outside_weights = rpn_outs[4]

        labels_int64 = fluid.layers.cast(x=labels_int32, dtype='int64')
        labels_int64.stop_gradient = True
        loss_cls_batch = fluid.layers.softmax_with_cross_entropy(
            logits=cls_score, label=labels_int64, numeric_stable_mode=True)
        loss_bbox_batch = fluid.layers.smooth_l1(
            x=bbox_pred,
            y=bbox_targets,
            inside_weight=bbox_inside_weights,
            outside_weight=bbox_outside_weights,
            sigma=1.0)
        loss_cls_batch.stop_gradient = True
        loss_bbox_batch.stop_gradient = True
        loss = loss_cls_batch + loss_bbox_batch

        topk_idx = fluid.default_main_program().current_block().create_var(
            dtype=bbox_outside_weights.dtype, shape=[-1])

        def get_param():
            return self.batch_size_per_im

        def sample_roi(loss_batch):
            batch_size_per_im = get_param()
            lod_info = loss_batch.lod()[0]
            lod_info_sampled = [0]

            total_samples = (len(lod_info) - 1) * batch_size_per_im
            idx_sorted = np.argsort(loss_batch, axis=0)
            topk_idx = np.squeeze(idx_sorted)[-total_samples:]

            return topk_idx

        topk_idx = fluid.layers.py_func(func=sample_roi, x=loss, out=topk_idx)
        return topk_idx
