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

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Constant
from paddle.fluid.regularizer import L2Decay

from ..registry import RetinaHeads

__all__ = ['RetinaHead']


@RetinaHeads.register
class RetinaHead(object):
    """
    RetinaHead class
    Args:
        cfg(dict): All parameters in dictionary
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.k_max = cfg.FPN.RPN_MAX_LEVEL
        self.k_min = cfg.FPN.RPN_MIN_LEVEL
        self.fpn_dim = cfg.FPN.DIM

    def _class_subnet(self, body_feats, spatial_scale, fpn_name_list):
        """
        Get class predictions of all level FPN level.
        
        Args:
            fpn_dict(Dict): A dictionary represents the output of FPN neck with 
                their name.
            spatial_scale(List): A list of multiplicative spatial scale factor.
            fpn_name_list(List): A list of names regarding to output of FPN neck.

        Return:
            cls_pred_input(List): Class prediction of all input fpn levels.
        """
        retina_cfg = self.cfg.RETINA_HEAD
        assert len(body_feats) == self.k_max - self.k_min + 1
        cls_pred_list = []
        for lvl in range(self.k_min, self.k_max + 1):
            fpn_name = fpn_name_list[self.k_max - lvl]
            subnet_blob = body_feats[fpn_name]
            for i in range(retina_cfg.NUM_CONVS):
                conv_name = 'retnet_cls_conv_n{}_fpn{}'.format(i, lvl)
                conv_share_name = 'retnet_cls_conv_n{}_fpn{}'.format(i,
                                                                     self.k_min)
                subnet_blob_in = subnet_blob
                subnet_blob = fluid.layers.conv2d(
                    input=subnet_blob_in,
                    num_filters=self.fpn_dim,
                    filter_size=3,
                    stride=1,
                    padding=1,
                    act='relu',
                    name=conv_name,
                    param_attr=ParamAttr(
                        name=conv_share_name + '_w',
                        initializer=Normal(
                            loc=0., scale=0.01)),
                    bias_attr=ParamAttr(
                        name=conv_share_name + '_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.)))

            # class prediction
            cls_name = 'retnet_cls_pred_fpn{}'.format(lvl)
            cls_share_name = 'retnet_cls_pred_fpn{}'.format(self.k_min)
            num_anchors = retina_cfg.SCALES_PER_OCTAVE * len(
                retina_cfg.ASPECT_RATIOS)
            cls_dim = num_anchors * (self.cfg.DATA.CLASS_NUM - 1)
            # bias initialization: b = -log((1 - pai) / pai)
            bias_init = float(-np.log((1 - retina_cfg.TRAIN.PRIOR_PROB) /
                                      retina_cfg.TRAIN.PRIOR_PROB))
            out_cls = fluid.layers.conv2d(
                input=subnet_blob,
                num_filters=cls_dim,
                filter_size=3,
                stride=1,
                padding=1,
                act=None,
                name=cls_name,
                param_attr=ParamAttr(
                    name=cls_share_name + '_w',
                    initializer=Normal(
                        loc=0., scale=0.01)),
                bias_attr=ParamAttr(
                    name=cls_share_name + '_b',
                    initializer=Constant(value=bias_init),
                    learning_rate=2.,
                    regularizer=L2Decay(0.)))
            cls_pred_list.append(out_cls)
        return cls_pred_list

    def _bbox_subnet(self, body_feats, spatial_scale, fpn_name_list):
        """
        Get bounding box predictions of all level FPN level.
        
        Args:
            fpn_dict(Dict): A dictionary represents the output of FPN neck with 
                their name.
            spatial_scale(List): A list of multiplicative spatial scale factor.
            fpn_name_list(List): A list of names regarding to output of FPN neck.

        Return:
            bbox_pred_input(List): Bounding box prediction of all input fpn
                levels.
        """
        retina_cfg = self.cfg.RETINA_HEAD
        assert len(body_feats) == self.k_max - self.k_min + 1
        bbox_pred_list = []
        for lvl in range(self.k_min, self.k_max + 1):
            fpn_name = fpn_name_list[self.k_max - lvl]
            subnet_blob = body_feats[fpn_name]
            for i in range(retina_cfg.NUM_CONVS):
                conv_name = 'retnet_bbox_conv_n{}_fpn{}'.format(i, lvl)
                conv_share_name = 'retnet_bbox_conv_n{}_fpn{}'.format(
                    i, self.k_min)
                subnet_blob_in = subnet_blob
                subnet_blob = fluid.layers.conv2d(
                    input=subnet_blob_in,
                    num_filters=self.fpn_dim,
                    filter_size=3,
                    stride=1,
                    padding=1,
                    act='relu',
                    name=conv_name,
                    param_attr=ParamAttr(
                        name=conv_share_name + '_w',
                        initializer=Normal(
                            loc=0., scale=0.01)),
                    bias_attr=ParamAttr(
                        name=conv_share_name + '_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.)))

            # bbox prediction
            bbox_name = 'retnet_bbox_pred_fpn{}'.format(lvl)
            bbox_share_name = 'retnet_bbox_pred_fpn{}'.format(self.k_min)
            num_anchors = retina_cfg.SCALES_PER_OCTAVE * len(
                retina_cfg.ASPECT_RATIOS)
            bbox_dim = num_anchors * 4
            out_bbox = fluid.layers.conv2d(
                input=subnet_blob,
                num_filters=bbox_dim,
                filter_size=3,
                stride=1,
                padding=1,
                act=None,
                name=bbox_name,
                param_attr=ParamAttr(
                    name=bbox_share_name + '_w',
                    initializer=Normal(
                        loc=0., scale=0.01)),
                bias_attr=ParamAttr(
                    name=bbox_share_name + '_b',
                    learning_rate=2.,
                    regularizer=L2Decay(0.)))
            bbox_pred_list.append(out_bbox)
        return bbox_pred_list

    def _anchor_generate(self, body_feats, spatial_scale, fpn_name_list):
        """
        Get anchor boxes of all level FPN level.
        
        Args:
            fpn_dict(Dict): A dictionary represents the output of FPN neck with 
                their name.
            spatial_scale(List): A list of multiplicative spatial scale factor.
            fpn_name_list(List): A list of names regarding to output of FPN neck.

        Return:
            anchor_input(List): Anchors of all input fpn levels with shape of.
            anchor_var_input(List): Anchor variance of all input fpn levels with
                shape.
        """
        retina_cfg = self.cfg.RETINA_HEAD
        assert len(body_feats) == self.k_max - self.k_min + 1
        anchor_list = []
        anchor_var_list = []
        for lvl in range(self.k_min, self.k_max + 1):
            anchor_sizes = []
            stride = int(1 / spatial_scale[self.k_max - lvl])
            for octave in range(retina_cfg.SCALES_PER_OCTAVE):
                anchor_size = stride * (2**(float(octave) /
                                            float(retina_cfg.SCALES_PER_OCTAVE))
                                        ) * retina_cfg.ANCHOR_SCALE
                anchor_sizes.append(anchor_size)
            fpn_name = fpn_name_list[self.k_max - lvl]
            anchor, anchor_var = fluid.layers.anchor_generator(
                input=body_feats[fpn_name],
                anchor_sizes=anchor_sizes,
                aspect_ratios=retina_cfg.ASPECT_RATIOS,
                variance=retina_cfg.VARIANCE,
                stride=[stride, stride])
            anchor_list.append(anchor)
            anchor_var_list.append(anchor_var)
        return anchor_list, anchor_var_list

    def _get_output(self, body_feats, spatial_scale, fpn_name_list):
        """
        Get class, bounding box predictions and anchor boxes of all level FPN level.
        
        Args:
            fpn_dict(Dict): A dictionary represents the output of FPN neck with 
                their name.
            spatial_scale(List): A list of multiplicative spatial scale factor.
            fpn_name_list(List): A list of names regarding to output of FPN neck.

        Return:
            cls_pred_input(List): Class prediction of all input fpn levels.
            bbox_pred_input(List): Bounding box prediction of all input fpn
                levels.
            anchor_input(List): Anchors of all input fpn levels with shape of.
            anchor_var_input(List): Anchor variance of all input fpn levels with
                shape.
        """
        retina_cfg = self.cfg.RETINA_HEAD
        assert len(body_feats) == self.k_max - self.k_min + 1
        # class subnet
        cls_pred_list = self._class_subnet(body_feats, spatial_scale,
                                           fpn_name_list)
        # bbox subnet
        bbox_pred_list = self._bbox_subnet(body_feats, spatial_scale,
                                           fpn_name_list)
        #generate anchors
        anchor_list, anchor_var_list = self._anchor_generate(
            body_feats, spatial_scale, fpn_name_list)

        cls_pred_reshape_list = []
        bbox_pred_reshape_list = []
        anchor_reshape_list = []
        anchor_var_reshape_list = []
        for i in range(self.k_max - self.k_min + 1):
            cls_pred_transpose = fluid.layers.transpose(
                cls_pred_list[i], perm=[0, 2, 3, 1])
            cls_pred_reshape = fluid.layers.reshape(
                cls_pred_transpose, shape=(0, -1, self.cfg.DATA.CLASS_NUM - 1))
            bbox_pred_transpose = fluid.layers.transpose(
                bbox_pred_list[i], perm=[0, 2, 3, 1])
            bbox_pred_reshape = fluid.layers.reshape(
                bbox_pred_transpose, shape=(0, -1, 4))
            anchor_reshape = fluid.layers.reshape(anchor_list[i], shape=(-1, 4))
            anchor_var_reshape = fluid.layers.reshape(
                anchor_var_list[i], shape=(-1, 4))
            cls_pred_reshape_list.append(cls_pred_reshape)
            bbox_pred_reshape_list.append(bbox_pred_reshape)
            anchor_reshape_list.append(anchor_reshape)
            anchor_var_reshape_list.append(anchor_var_reshape)
        output = {}
        output['cls_pred'] = cls_pred_reshape_list
        output['bbox_pred'] = bbox_pred_reshape_list
        output['anchor'] = anchor_reshape_list
        output['anchor_var'] = anchor_var_reshape_list
        return output

    def get_prediction(self, body_feats, spatial_scale, fpn_name_list, im_info):
        """
        Get prediction bounding box in test stage.
        
        Args:
            fpn_dict(Dict): A dictionary represents the output of FPN neck with 
                their name.
            spatial_scale(List): A list of multiplicative spatial scale factor.
            fpn_name_list(List): A list of names regarding to output of FPN neck.
            im_info (Variable): A 2-D LoDTensor with shape [B, 3]. B is the 
                number of input images, each element consists of im_height, 
                im_width, im_scale.
            cls_score (Variable), bbox_pred(Variable): Output of get_output.
            im_info(Variable): A 2-D LoDTensor with shape [B, 3]. B is the 
                number of input images, each element consists of im_height, 
                im_width, im_scale.
     
        Returns:
            pred_result(Variable): Prediction result with shape [N, 6]. Each 
                row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]. 
                N is the total number of prediction.
        """
        output = self._get_output(body_feats, spatial_scale, fpn_name_list)
        cls_pred_reshape_list = output['cls_pred']
        bbox_pred_reshape_list = output['bbox_pred']
        anchor_reshape_list = output['anchor']
        anchor_var_reshape_list = output['anchor_var']
        for i in range(self.k_max - self.k_min + 1):
            cls_pred_reshape_list[i] = fluid.layers.sigmoid(
                cls_pred_reshape_list[i])
        pred_result = fluid.layers.retinanet_detection_output(
            bboxes=bbox_pred_reshape_list,
            scores=cls_pred_reshape_list,
            anchors=anchor_reshape_list,
            im_info=im_info,
            score_threshold=self.cfg.RETINA_HEAD.TEST.SCORE_THRESH,
            nms_top_k=self.cfg.RETINA_HEAD.TEST.PRE_NMS_TOP_N,
            keep_top_k=self.cfg.RETINA_HEAD.TEST.DETECTIONS_PER_IM,
            nms_threshold=self.cfg.RETINA_HEAD.TEST.NMS_THRESH,
            nms_eta=self.cfg.RETINA_HEAD.TEST.NMA_ETA)
        return {'bbox': pred_result}

    def get_loss(self, body_feats, spatial_scale, fpn_name_list, im_info,
                 gt_box, gt_label, is_crowd):
        """
        Calculate the loss of retinanet.
        Args:
            fpn_dict(Dict): A dictionary represents the output of FPN neck with 
                their name.
            spatial_scale(List): A list of multiplicative spatial scale factor.
            fpn_name_list(List): A list of names regarding to output of FPN neck.
            im_info(Variable): A 2-D LoDTensor with shape [B, 3]. B is the 
                number of input images, each element consists of im_height, 
                im_width, im_scale.
            gt_box(Variable): The ground-truth bounding boxes with shape [M, 4].
                M is the number of groundtruth.
            gt_label(Variable): The ground-truth labels with shape [M, 1].
                M is the number of groundtruth.
            is_crowd(Variable): Indicates groud-truth is crowd or not with
                shape [M, 1]. M is the number of groundtruth.

        Returns:
            Type: Dict
                loss_cls(Variable): focal loss.
                loss_bbox(Variable): smooth l1 loss.
        """
        output = self._get_output(body_feats, spatial_scale, fpn_name_list)
        cls_pred_reshape_list = output['cls_pred']
        bbox_pred_reshape_list = output['bbox_pred']
        anchor_reshape_list = output['anchor']
        anchor_var_reshape_list = output['anchor_var']

        cls_pred_input = fluid.layers.concat(cls_pred_reshape_list, axis=1)
        bbox_pred_input = fluid.layers.concat(bbox_pred_reshape_list, axis=1)
        anchor_input = fluid.layers.concat(anchor_reshape_list, axis=0)
        anchor_var_input = fluid.layers.concat(anchor_var_reshape_list, axis=0)

        score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight, fg_num = \
            fluid.layers.retinanet_target_assign(
                bbox_pred=bbox_pred_input,
                cls_logits=cls_pred_input,
                anchor_box=anchor_input,
                anchor_var=anchor_var_input,
                gt_boxes=gt_box,
                gt_labels=gt_label,
                is_crowd=is_crowd,
                im_info=im_info,
                num_classes=self.cfg.DATA.CLASS_NUM - 1,
                positive_overlap=self.cfg.RETINA_HEAD.TRAIN.POSITIVE_OVERLAP,
                negative_overlap=self.cfg.RETINA_HEAD.TRAIN.NEGATIVE_OVERLAP)

        fg_num = fluid.layers.reduce_sum(fg_num, name='fg_num')
        loss_cls = fluid.layers.sigmoid_focal_loss(
            x=score_pred,
            label=score_tgt,
            fg_num=fg_num,
            gamma=self.cfg.RETINA_HEAD.TRAIN.LOSS_GAMMA,
            alpha=self.cfg.RETINA_HEAD.TRAIN.LOSS_ALPHA)
        loss_cls = fluid.layers.reduce_sum(loss_cls, name='loss_cls')
        loss_bbox = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=self.cfg.RETINA_HEAD.TRAIN.SIGMA,
            inside_weight=bbox_weight,
            outside_weight=bbox_weight)
        loss_bbox = fluid.layers.reduce_sum(loss_bbox, name='loss_bbox')
        loss_bbox = loss_bbox / fg_num

        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox}
