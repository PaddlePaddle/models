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
from paddle.fluid.initializer import Normal, Xavier
from paddle.fluid.regularizer import L2Decay

from ..registry import BBoxHeads
from ..registry import BBoxHeadConvs

__all__ = ['BBoxHead']


@BBoxHeads.register
class BBoxHead(object):
    """
    BBoxHead class

    Args:
        cfg (Dict): All parameters in dictionary.

    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.class_num = self.cfg.DATA.CLASS_NUM
        self.head_func = BBoxHeadConvs.get(cfg.BBOX_HEAD.HEAD_CONV)(cfg)

    def _get_output(self, roi_feat):
        """
        Get bbox head output.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.

        Returns:
            cls_score(Variable): Output of rpn head with shape of 
                [N, num_anchors, H, W].
            bbox_pred(Variable): Output of rpn head with shape of
                [N, num_anchors * 4, H, W].
        """
        head_feat = self.head_func(roi_feat)
        cls_score = fluid.layers.fc(input=head_feat,
                                    size=self.class_num,
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
                                    size=4 * self.class_num,
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

    def get_loss(self, roi_feat, labels_int32, bbox_targets,
                 bbox_inside_weights, bbox_outside_weights):
        """
        Get bbox_head loss.
        
        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.
            labels_int32(Variable): Class label of a RoI with shape [P, 1].
                P is the number of RoI.
            bbox_targets(Variable): Box label of a RoI with shape 
                [P, 4 * class_nums].
            bbox_inside_weights(Variable): Indicates whether a box should 
                contribute to loss. Same shape as bbox_targets.
            bbox_outside_weights(Variable): Indicates whether a box should 
                contribute to loss. Same shape as bbox_targets.

        Return:
            Type: Dict
                loss_cls(Variable): bbox_head loss.
                loss_bbox(Variable): bbox_head loss.
        """

        cls_score, bbox_pred = self._get_output(roi_feat)

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
        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox}

    def get_prediction(self, roi_feat, rois, im_info):
        """
        Get prediction bounding box in test stage.
        
        Args:
            rois (Variable): Output of generate_proposals in rpn head.
            im_info (Variable): A 2-D LoDTensor with shape [B, 3]. B is the 
                number of input images, each element consists of im_height, 
                im_width, im_scale.
            cls_score (Variable), bbox_pred(Variable): Output of get_output.
     
        Returns:
            pred_result(Variable): Prediction result with shape [N, 6]. Each 
                row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]. 
                N is the total number of prediction.
        """
        cls_score, bbox_pred = self._get_output(roi_feat)

        im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
        im_scale_lod = fluid.layers.sequence_expand(im_scale, rois)
        boxes = rois / im_scale_lod
        cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)
        bbox_pred = fluid.layers.reshape(bbox_pred, (-1, self.class_num, 4))
        decoded_box = fluid.layers.box_coder(
            prior_box=boxes,
            prior_box_var=self.cfg.RPN_HEAD.PROPOSAL.BBOX_REG_WEIGHTS,
            target_box=bbox_pred,
            code_type='decode_center_size',
            box_normalized=False,
            axis=1)
        cliped_box = fluid.layers.box_clip(input=decoded_box, im_info=im_info)
        pred_result = fluid.layers.multiclass_nms(
            bboxes=cliped_box,
            scores=cls_prob,
            score_threshold=self.cfg.TEST.SCORE_THRESH,
            nms_top_k=-1,
            nms_threshold=self.cfg.TEST.NMS_THRESH,
            keep_top_k=self.cfg.TEST.DETECTIONS_PER_IM,
            normalized=False)
        return {'bbox': pred_result}


@BBoxHeadConvs.register
class BBox2MLP(object):
    def __init__(self, cfg):
        self.mlp_head_dim = cfg.BBOX_HEAD.MLP_HEAD_DIM

    def __call__(self, roi_feat):
        fc6 = fluid.layers.fc(input=roi_feat,
                              size=self.mlp_head_dim,
                              act='relu',
                              name='fc6',
                              param_attr=ParamAttr(
                                  name='fc6_w', initializer=Xavier()),
                              bias_attr=ParamAttr(
                                  name='fc6_b',
                                  learning_rate=2.,
                                  regularizer=L2Decay(0.)))
        head_feat = fluid.layers.fc(input=fc6,
                                    size=self.mlp_head_dim,
                                    act='relu',
                                    name='fc7',
                                    param_attr=ParamAttr(
                                        name='fc7_w', initializer=Xavier()),
                                    bias_attr=ParamAttr(
                                        name='fc7_b',
                                        learning_rate=2.,
                                        regularizer=L2Decay(0.)))
        return head_feat
