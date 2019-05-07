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
import ppdet.models.backbones as backbones
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Xavier
from paddle.fluid.regularizer import L2Decay


class BBoxHead(object):
    """
    BBoxHead class
    """

    def __init__(self,
                 cfg,
                 gt_label,
                 is_crowd,
                 gt_box,
                 im_info,
                 head_func,
                 use_random=True):
        self.cfg = cfg
        self.gt_label = gt_label
        self.is_crowd = is_crowd
        self.gt_box = gt_box
        self.im_info = im_info
        self.head_func = head_func
        self.use_random = use_random

    def _get_head_feat(self, roi_feat):
        """
        Get head feature from RoI feature. Two head function are avaiable:
        bboxC5_head and bbox2mlp_head.
        
        Args:
            roi_feat: RoI feature from RoIExtractor.
        
        Returns:
            head_feat: head feature, used for bbox_head output.
        """
        head_feat = self.head_func(self.cfg, roi_feat)
        return head_feat

    def get_output(self, roi_feat):
        """
        Get bbox head output.

        Args:
            roi_feat: RoI feature from RoIExtractor.

        Returns:
            cls_score, bbox_pred
        """
        class_num = self.cfg.DATA.CLASS_NUM
        head_feat = self._get_head_feat(roi_feat)
        cls_score = fluid.layers.fc(input=head_feat,
                                    size=class_num,
                                    act=None,
                                    name='cls_score',
                                    param_attr=ParamAttr(
                                        name='cls_score_w',
                                        initializer=Normal(
                                            loc=0.0, scale=0.01)),
                                    bias_attr=ParamAttr(
                                        name='cls_score_b',
                                        learning_rate=2.,
                                        regularizer=L2Decay(0.)))
        bbox_pred = fluid.layers.fc(input=head_feat,
                                    size=4 * class_num,
                                    act=None,
                                    name='bbox_pred',
                                    param_attr=ParamAttr(
                                        name='bbox_pred_w',
                                        initializer=Normal(
                                            loc=0.0, scale=0.001)),
                                    bias_attr=ParamAttr(
                                        name='bbox_pred_b',
                                        learning_rate=2.,
                                        regularizer=L2Decay(0.)))
        return cls_score, bbox_pred

    def get_target(self, rpn_rois):
        """
        Get bbox target for bbox head loss

        Args:
            rpn_rois: Output of generate_proposals in rpn head

        Returns:
            rois: RoI with shape [P, 4]. P is usually equal to  
                  batch_size_per_im * batch_size, each element 
                  is a bounding box with [xmin, ymin, xmax, ymax] format.
            labels_int32: Class label of a RoI with shape [P, 1]. 
            bbox_targets: Box label of a RoI with shape [P, 4 * class_nums].
            bbox_inside_weights: Indicates whether a box should contribute 
                                 to loss. Same shape as bbox_targets.
            bbox_outside_weights: Indicates whether a box should contribute
                                 to loss. Same shape as bbox_targets.
        """
        cfg = self.cfg
        outs = fluid.layers.generate_proposal_labels(
            rpn_rois=rpn_rois,
            gt_classes=self.gt_label,
            is_crowd=self.is_crowd,
            gt_boxes=self.gt_box,
            im_info=self.im_info,
            batch_size_per_im=cfg.BBOX_HEAD.ROI.BATCH_SIZE_PER_IM,
            fg_fraction=cfg.BBOX_HEAD.ROI.FG_FRACTION,
            fg_thresh=cfg.BBOX_HEAD.ROI.FG_THRESH,
            bg_thresh_hi=cfg.BBOX_HEAD.ROI.BG_THRESH_HIGH,
            bg_thresh_lo=cfg.BBOX_HEAD.ROI.BG_THRESH_LOW,
            bbox_reg_weights=cfg.BBOX_HEAD.ROI.BBOX_REG_WEIGHTS,
            class_nums=cfg.DATA.CLASS_NUM,
            use_random=self.use_random)

        rois = outs[0]
        labels_int32 = outs[1]
        bbox_targets = outs[2]
        bbox_inside_weights = outs[3]
        bbox_outside_weights = outs[4]
        return rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def get_loss(self, cls_score, bbox_pred, labels_int32, bbox_targets,
                 bbox_inside_weights, bbox_outside_weights):
        """
        Get bbox_head loss.
        """
        labels_int64 = fluid.layers.cast(x=labels_int32, dtype='int64')
        labels_int64.stop_gradient = True
        loss_cls = fluid.layers.softmax_with_cross_entropy(
            logits=cls_score, label=labels_int64, numeric_stable_mode=True)
        loss_cls = fluid.layers.reduce_mean(loss_cls)
        loss_bbox = fluid.layers.smooth_l1(
            x=bbox_pred,
            y=bbox_targets,
            inside_weight=bbox_inside_weights,
            outside_weight=bbox_outside_weights,
            sigma=1.0)
        loss_bbox = fluid.layers.reduce_mean(loss_bbox)
        return loss_cls, loss_bbox

    def get_prediction(
            self,
            rpn_rois,
            cls_score,
            bbox_pred, ):
        """
        Get prediction bounding box in test stage.
        
        Args:
            rpn_rois: Output of generate_proposals in rpn head.
            cls_score, bbox_pred: Output of get_output.
     
        Returns:
            pred_result: Prediction result with shape [N, 6]. Each row has 6 
                         values: [label, confidence, xmin, ymin, xmax, ymax]. 
                         N is the total number of prediction.
        """
        class_num = self.cfg.DATA.CLASS_NUM
        im_scale = fluid.layers.slice(self.im_info, [1], starts=[2], ends=[3])
        im_scale_lod = fluid.layers.sequence_expand(im_scale, rpn_rois)
        boxes = rpn_rois / im_scale_lod
        cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)
        bbox_pred_reshape = fluid.layers.reshape(bbox_pred, (-1, class_num, 4))
        decoded_box = fluid.layers.box_coder(
            prior_box=boxes,
            prior_box_var=self.cfg.BBOX_HEAD.ROI.BBOX_REG_WEIGHTS,
            target_box=bbox_pred_reshape,
            code_type='decode_center_size',
            box_normalized=False,
            axis=1)
        cliped_box = fluid.layers.box_clip(
            input=decoded_box, im_info=self.im_info)
        pred_result = fluid.layers.multiclass_nms(
            bboxes=cliped_box,
            scores=cls_prob,
            score_threshold=self.cfg.TEST.SCORE_THRESH,
            nms_top_k=-1,
            nms_threshold=self.cfg.TEST.NMS_THRESH,
            keep_top_k=self.cfg.TEST.DETECTIONS_PER_IM,
            normalized=False)
        return pred_result


def bboxC5_head(cfg, roi_feat):
    res5_2_sum = backbones.resnet.ResNet50C5(roi_feat)
    head_feat = fluid.layers.pool2d(
        res5_2_sum, pool_type='avg', pool_size=7, name='res5_pool')
    return head_feat


def bbox2mlp_head(cfg, roi_feat):
    fc6 = fluid.layers.fc(input=roi_feat,
                          size=cfg.BBOX_HEAD.MLP_HEAD_DIM,
                          act='relu',
                          name='fc6',
                          param_attr=ParamAttr(
                              name='fc6_w', initializer=Xavier()),
                          bias_attr=ParamAttr(
                              name='fc6_b',
                              learning_rate=2.,
                              regularizer=L2Decay(0.)))
    head_feat = fluid.layers.fc(input=fc6,
                                size=cfg.BBOX_HEAD.MLP_HEAD_DIM,
                                act='relu',
                                name='fc7',
                                param_attr=ParamAttr(
                                    name='fc7_w', initializer=Xavier()),
                                bias_attr=ParamAttr(
                                    name='fc7_b',
                                    learning_rate=2.,
                                    regularizer=L2Decay(0.)))
    return head_feat
