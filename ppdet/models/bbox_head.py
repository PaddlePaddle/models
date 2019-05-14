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
        """
        Args:
            cfg(Dict): All parameters in dictionary.
            gt_label(Variable): Class label of groundtruth with shape [M, 1].
                                M is the number of groundtruth.
            is_crowd(Variable): A flag indicates whether a groundtruth is crowd
                                with shape [M, 1]. M is the number of 
                                groundtruth.
            gt_box(Variable): Bounding box in [xmin, ymin, xmax, ymax] format 
                              with shape [M, 4]. M is the number of groundtruth.
            im_info(Variable): A 2-D LoDTensor with shape [B, 3]. B is the 
                               number of input images, each element consists 
                               of im_height, im_width, im_scale.
            head_func(Function): A function to extract bbox_head feature.
            use_random(bool): Use random sampling to choose foreground and 
                              background boxes.
        """
        self.cfg = cfg
        self.gt_label = gt_label
        self.is_crowd = is_crowd
        self.gt_box = gt_box
        self.im_info = im_info
        self.head_func = head_func
        self.use_random = use_random

    def get_output(self, roi_feat):
        """
        Get bbox head output.

        Args:
            roi_feat(Variable): RoI feature from RoIExtractor.

        Returns:
            cls_score(Variable), bbox_pred(Variable) are output of bbox_head.
        """
        class_num = self.cfg.DATA.CLASS_NUM
        head_feat = self.head_func(self.cfg, roi_feat)
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
            rpn_rois(Variable): Output of generate_proposals in rpn head.

        Returns:
            rois(Variable): RoI with shape [P, 4]. P is usually equal to  
                            batch_size_per_im * batch_size, each element 
                            is a bounding box with [xmin, ymin, xmax, ymax] 
                            format.
            labels_int32(Variable): Class label of a RoI with shape [P, 1]. 
            bbox_targets(Variable): Box label of a RoI with shape 
                                    [P, 4 * class_nums].
            bbox_inside_weights(Variable): Indicates whether a box should 
                                           contribute to loss. Same shape as
                                           bbox_targets.
            bbox_outside_weights(Variable): Indicates whether a box should 
                                            contribute to loss. Same shape as 
                                            bbox_targets.
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
        
        Args:
            cls_score(Variable), bbox_pred(Variable), labels_int32(Variable),
            bbox_targets(Variable), bbox_inside_weights(Variable), 
            bbox_outside_weights(Variable) are outputs of get_target.

        Return:
            loss_cls(Variable): bbox_head loss.
            loss_bbox(Variable): bbox_head loss.
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
            rpn_rois(Variable): Output of generate_proposals in rpn head.
            cls_score(Variable), bbox_pred(Variable): Output of get_output.
     
        Returns:
            pred_result(Variable): Prediction result with shape [N, 6]. Each 
               row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]. 
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
