#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import MSRA

from ..registry import SSDHeads

__all__ = ['SSDHead']


@SSDHeads.register
class SSDHead(object):
    """
        SSDHead class
    """
    def __init__(self, cfg):
        """
        Args:
            cfg(dict): All parameters in dictionary
        """
        self.cfg = cfg

    def _get_output(self, input1, input2):
        """
        Get the SSD output.
        Args:
            input1 (Variable): feature map from backbone with
                                shape of [N, C, H, W]
            input2 (Variable): feature map from backbone with
                                shape of [N, C, H, W]

        Returns:
            mbox_locs (Variable): The predicted boxes’ location of the inputs.
                                  The layout is [N, H*W*num_priors, 4].
            mbox_confs (Variable): The predicted boxes’ confidence
                                   of the inputs.
                                   The layout is [N, H*W*num_priors, C].
            box (Variable): The output prior boxes of PriorBox.
                            The layout is [num_priors, 4].
            box_var (Variable): The expanded variances of PriorBox.
                                The layout is [num_priors, 4].

        tips：The num_priors is the number of predicted boxes
              each position of each input
              and C is the number of Classes.
        """
        def _conv_norm(input,
                       filter_size,
                       num_filters,
                       stride,
                       padding,
                       channels=None,
                       num_groups=1,
                       act='relu',
                       use_cudnn=True):
            parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
            conv = fluid.layers.conv2d(
                input=input,
                num_filters=num_filters,
                filter_size=filter_size,
                stride=stride,
                padding=padding,
                groups=num_groups,
                act=None,
                use_cudnn=use_cudnn,
                param_attr=parameter_attr,
                bias_attr=False)
            return fluid.layers.batch_norm(input=conv, act=act)

        def extra_block(input, num_filters1, num_filters2, num_groups, scale):
            '''
            Get the feature map which is used to get bbox and label.
            Contain two convolution process, so there are two filters.
            '''
            # 1x1 conv
            pointwise_conv = _conv_norm(
                input=input,
                filter_size=1,
                num_filters=int(num_filters1 * scale),
                stride=1,
                num_groups=int(num_groups * scale),
                padding=0)
            # 3x3 conv
            normal_conv = _conv_norm(
                input=pointwise_conv,
                filter_size=3,
                num_filters=int(num_filters2 * scale),
                stride=2,
                num_groups=int(num_groups * scale),
                padding=1)
            return normal_conv

        # 10x10
        scale = self.cfg.MODEL.CONV_GROUP_SCALE
        module14 = extra_block(input2, 256, 512, 1, scale)
        # 5x5
        module15 = extra_block(module14, 128, 256, 1, scale)
        # 3x3
        module16 = extra_block(module15, 128, 256, 1, scale)
        # 2x2
        module17 = extra_block(module16, 64, 128, 1, scale)
        mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
            inputs=[input1, input2, module14, module15, module16, module17],
            image=self.img,
            num_classes=self.cfg.DATA.CLASS_NUM,
            min_ratio=self.cfg.SSD_HEAD.MIN_RATIO,
            max_ratio=self.cfg.SSD_HEAD.MAX_RATIO,
            min_sizes=self.cfg.SSD_HEAD.MIN_SIZES,
            max_sizes=self.cfg.SSD_HEAD.MAX_SIZES,
            aspect_ratios=self.cfg.SSD_HEAD.ASPECT_RATIOS,
            base_size=self.cfg.TRAIN.SHAPE,
            offset=self.cfg.SSD_HEAD.CENTER_OFFSET,
            flip=self.cfg.SSD_HEAD.FLIP_ASPECT)
        return mbox_locs, mbox_confs, box, box_var

    def get_prediction(self, input, body_feat1, body_feat2):
        """
        Get prediction bbox and label according to the output of ssd head.
        Args:
            body_feat1 (Variable): feature map from backbone
                                   with shape of [N, C, H, W]
            body_feat2 (Variable): feature map from backbone
                                   with shape of [N, C, H, W]
        Returns:
            nmsed_out (Variable): The detection outputs is a LoDTensor with shape [count, 6].
                        Each row has six values: [label, confidence, xmin, ymin, xmax, ymax].
        """
        self.img = input
        locs, confs, box, box_var = self._get_output(body_feat1, body_feat2)
        nms_thresh = self.cfg.SSD_HEAD.NMS_THRESH
        self.nmsed_out = fluid.layers.detection_output(
            locs, confs, box, box_var, nms_threshold=nms_thresh)
        return self.nmsed_out
        
    def get_loss(self, input, body_feat1, body_feat2, gt_bbox, gt_label):
        """
        Calculate the loss of prediction result and real result.
        Args:
            gt_box (Variable): The ground-truth bounding boxes.
            gt_label (Variable): The ground-truth labels.
        Returns:
            loss (Variable): Return in train process.
                             The sum of ssd loss.
        """
        self.img = input
        locs, confs, box, box_var = self._get_output(body_feat1, body_feat2)
        loss = fluid.layers.ssd_loss(locs, confs,
                                gt_bbox, gt_label,
                                box, box_var)
        loss = fluid.layers.reduce_sum(loss)
        return loss
    
    def get_map(self, gt_bbox, gt_label):
        """
        Calculate the map of prediction result and real result.
        Args:
            body_feat1 (Variable): feature map from backbone with shape of [N, C, H, W]
            body_feat2 (Variable): feature map from backbone with shape of [N, C, H, W]
            gt_box (Variable): The ground-truth bounding boxes.
            gt_label (Variable): The ground-truth labels.
        Returns:
            map_eval (Variable): Return in verification or inference process.
        """
        map_eval = fluid.metrics.DetectionMAP(
            self.nmsed_out,
            gt_label,
            gt_bbox,
            None,
            self.cfg.DATA.CLASS_NUM,
            overlap_threshold=self.cfg.METRIC.AP_OVERLAP_THRESH,
            evaluate_difficult=False,
            ap_version=self.cfg.METRIC.AP_VERSION)
        return map_eval
        
