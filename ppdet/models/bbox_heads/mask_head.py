#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay

from ..registry import MaskHeads

__all__ = ['MaskHead']


@MaskHeads.register
class MaskHead(object):
    """
    TODO(qingiqng): add more comments
    Args:
        cfg(Dict): All parameters in dictionary.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.is_train = getattr(cfg, 'IS_TRAIN', True)

    def _mask_conv_head(self, roi_feat, conv_num):
        for i in range(conv_num):
            layer_name = "mask_inter_feat_" + str(i + 1)
            print(i, layer_name)
            roi_feat = fluid.layers.conv2d(
                input=roi_feat,
                num_filters=self.cfg.MASK_HEAD.DIM_REDUCED,
                filter_size=3,
                padding=1 * self.cfg.MASK_HEAD.DILATION,
                act='relu',
                stride=1,
                dilation=self.cfg.MASK_HEAD.DILATION,
                name=layer_name,
                param_attr=ParamAttr(
                    name=layer_name + '_w', initializer=MSRA(uniform=True)),
                bias_attr=ParamAttr(
                    name=layer_name + '_b',
                    learning_rate=2.,
                    regularizer=L2Decay(0.)))
        feat = fluid.layers.conv2d_transpose(
            input=roi_feat,
            num_filters=self.cfg.MASK_HEAD.DIM_REDUCED,
            filter_size=2,
            stride=2,
            act='relu',
            param_attr=ParamAttr(
                name='conv5_mask_w', initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name='conv5_mask_b', learning_rate=2., regularizer=L2Decay(0.)))
        return feat

    def _get_output(self, roi_feat):
        class_num = self.cfg.DATA.CLASS_NUM
        # configure the conv number for FPN if necessary
        conv_num = 4 if getattr(self.cfg.MODEL, 'FPN', False) else 0
        head_feat = self._mask_conv_head(roi_feat, conv_num)
        mask_logits = fluid.layers.conv2d(
            input=head_feat,
            num_filters=class_num,
            filter_size=1,
            act=None,
            param_attr=ParamAttr(
                name='mask_fcn_logits_w', initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name="mask_fcn_logits_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        return mask_logits

    def get_loss(self, roi_feat, mask_int32):
        """
        TODO(qingqing): add more comments
        """
        mask_logits = self._get_output(roi_feat)
        class_num = self.cfg.DATA.CLASS_NUM
        resolution = self.cfg.MASK_HEAD.RESOLUTION
        dim = class_num * resolution * resolution
        mask_logits = fluid.layers.reshape(mask_logits, (-1, dim))

        mask_label = fluid.layers.cast(x=mask_int32, dtype='float32')
        mask_label.stop_gradient = True
        loss_mask = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=mask_logits, label=mask_label, ignore_index=-1, normalize=True)
        loss_mask = fluid.layers.reduce_sum(loss_mask, name='loss_mask')
        return {'loss_mask': loss_mask}

    def get_prediction(self, roi_feat, bbox_pred):
        """
        Get prediction mask in test stage.
        
        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.
            bbox_pred (Variable): predicted bbox.
    
        Returns:
            mask_pred (Variable): Prediction mask with shape
                [N, class_num, resolution, resolution].
        """
        mask_logits = self._get_output(roi_feat)
        mask_prob = fluid.layers.sigmoid(mask_logits)
        mask_prob = fluid.layers.lod_reset(mask_prob, bbox_pred)
        return mask_prob
