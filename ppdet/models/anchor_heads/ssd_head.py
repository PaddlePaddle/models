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
        Args:
            cfg(dict): All parameters in dictionary
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def _get_output(self, body_feats):
        """
        Get the SSD output.
        Args:
            body_feats (list): the list of feature map, which contains:
                input1 (Variable): the middle feature map of backbone
                                    with shape of [N, C, H, W]    
                input2 (Variable): output of backbone
                                    with shape of [N, C, H, W]
                input3~input6 (Variable): output of extra_layers
                                    with shape of [N, C, H, W]
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
        tips: The num_priors is the number of predicted boxes
              each position of each input
              and C is the number of Classes.
        """
        mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
            inputs=body_feats,
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

    def get_prediction(self, input, body_feats):
        """
        Get prediction bbox and label according to the output of ssd head.
        Args:
            input (Variable): the input image
            body_feats (list): the list of feature map, which contains:
                input1 (Variable): the middle feature map of backbone
                                    with shape of [N, C, H, W]    
                input2 (Variable): output of backbone
                                    with shape of [N, C, H, W]
                input3~input6 (Variable): output of extra_layers
                                    with shape of [N, C, H, W]
        Returns:
            nmsed_out (Variable): The detection outputs is a LoDTensor with shape [count, 6].
                        Each row has six values: [label, confidence, xmin, ymin, xmax, ymax].
        """
        self.img = input
        locs, confs, box, box_var = self._get_output(body_feats)
        nms_thresh = self.cfg.SSD_HEAD.NMS_THRESH
        self.nmsed_out = fluid.layers.detection_output(
            locs, confs, box, box_var, nms_threshold=nms_thresh)
        return {'bbox': self.nmsed_out}

    def get_loss(self, input, body_feats, gt_bbox, gt_label):
        """
        Calculate the loss of prediction result and real result.
        Args:
            input (Variable): the input image
            body_feats (list): the list of feature map, which contains:
                input1 (Variable): the middle feature map of backbone
                                    with shape of [N, C, H, W]    
                input2 (Variable): output of backbone
                                    with shape of [N, C, H, W]
                input3~input6 (Variable): output of extra_layers
                                    with shape of [N, C, H, W]
            gt_box (Variable): The ground-truth bounding boxes.
            gt_label (Variable): The ground-truth labels.
        Returns:
            loss (Variable): Return in train process.
                             The sum of ssd loss.
        """
        self.img = input
        locs, confs, box, box_var = self._get_output(body_feats)
        loss = fluid.layers.ssd_loss(locs, confs, gt_bbox, gt_label, box,
                                     box_var)
        loss = fluid.layers.reduce_sum(loss)
        return {'loss': loss}

    def get_map(self, gt_bbox, gt_label, difficult):
        """
        Calculate the map of prediction result and real result.
        Args:
            gt_box (Variable): The ground-truth bounding boxes.
            gt_label (Variable): The ground-truth labels.
        Returns:
            map_eval (Variable): Return in verification or inference process.
        """
        map_eval = fluid.metrics.DetectionMAP(
            self.nmsed_out,
            gt_label,
            gt_bbox,
            difficult,
            self.cfg.DATA.CLASS_NUM,
            overlap_threshold=self.cfg.METRIC.AP_OVERLAP_THRESH,
            evaluate_difficult=False,
            ap_version=self.cfg.METRIC.AP_VERSION)
        return {'map': map_eval}
