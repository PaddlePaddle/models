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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Normal
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay
from config import cfg
from models.ext_op.rrpn_lib import *


class RRPN(object):
    def __init__(self,
                 add_conv_body_func=None,
                 add_roi_box_head_func=None,
                 mode='train',
                 use_pyreader=True,
                 use_random=True):
        self.add_conv_body_func = add_conv_body_func
        self.add_roi_box_head_func = add_roi_box_head_func
        self.mode = mode
        self.use_pyreader = use_pyreader
        self.use_random = use_random

    def build_model(self):
        self.build_input()
        body_conv = self.add_conv_body_func(self.image)
        # RPN
        self.rpn_heads(body_conv)
        # Fast RCNN
        self.fast_rcnn_heads(body_conv)
        if self.mode != 'train':
            self.eval_bbox()

    def loss(self):
        losses = []
        # Fast RCNN loss
        loss_cls, loss_bbox = self.fast_rcnn_loss()
        # RPN loss
        rpn_cls_loss, rpn_reg_loss = self.rpn_loss()
        losses = [loss_cls, loss_bbox, rpn_cls_loss, rpn_reg_loss]
        rkeys = ['loss', 'loss_cls', 'loss_bbox', \
                 'loss_rpn_cls', 'loss_rpn_bbox',]
        loss = fluid.layers.sum(losses)
        rloss = [loss] + losses
        return rloss, rkeys, self.rpn_rois

    def eval_bbox_out(self):
        return self.pred_result

    def build_input(self):
        self.image = fluid.data(
            name='image', shape=[None, 3, None, None], dtype='float32')
        if self.mode == 'train':
            self.gt_box = fluid.data(
                name='gt_box', shape=[None, 5], dtype='float32', lod_level=1)
        else:
            self.gt_box = fluid.data(
                name='gt_box', shape=[None, 8], dtype='float32', lod_level=1)
        self.gt_label = fluid.data(
            name='gt_class', shape=[None, 1], dtype='int32', lod_level=1)
        self.is_crowd = fluid.data(
            name='is_crowed', shape=[None, 1], dtype='int32', lod_level=1)
        self.im_info = fluid.data(
            name='im_info', shape=[None, 3], dtype='float32')
        self.im_id = fluid.data(name='im_id', shape=[None, 1], dtype='int64')
        self.difficult = fluid.data(
            name='is_difficult', shape=[None, -1], dtype='float32', lod_level=1)
        if self.mode == 'train':
            feed_data = [
                self.image, self.gt_box, self.gt_label, self.is_crowd,
                self.im_info, self.im_id
            ]
        elif self.mode == 'infer':
            feed_data = [self.image, self.im_info]
        else:
            feed_data = [
                self.image, self.gt_box, self.gt_label, self.is_crowd,
                self.im_info, self.im_id, self.difficult
            ]
        if self.mode == 'train':
            self.data_loader = fluid.io.DataLoader.from_generator(
                feed_list=feed_data, capacity=64, iterable=False)
        else:
            self.data_loader = fluid.io.DataLoader.from_generator(
                feed_list=feed_data, capacity=64, iterable=True)

    def eval_bbox(self):
        self.im_scale = fluid.layers.slice(
            self.im_info, [1], starts=[2], ends=[3])
        im_scale_lod = fluid.layers.sequence_expand(self.im_scale,
                                                    self.rpn_rois)
        results = []
        boxes = self.rpn_rois
        cls_prob = fluid.layers.softmax(self.cls_score, use_cudnn=False)
        bbox_pred = fluid.layers.reshape(self.bbox_pred, (-1, cfg.class_num, 5))
        for i in range(cfg.class_num - 1):
            bbox_pred_slice = fluid.layers.slice(
                bbox_pred, axes=[1], starts=[i + 1], ends=[i + 2])
            bbox_pred_reshape = fluid.layers.reshape(bbox_pred_slice, (-1, 5))
            decoded_box = rrpn_box_coder(prior_box=boxes, \
                                         target_box=bbox_pred_reshape, \
                                         prior_box_var=cfg.bbox_reg_weights)
            score_slice = fluid.layers.slice(
                cls_prob, axes=[1], starts=[i + 1], ends=[i + 2])
            score_slice = fluid.layers.reshape(score_slice, shape=[-1, 1])
            box_positive = fluid.layers.reshape(decoded_box, shape=[-1, 8])
            box_reshape = fluid.layers.reshape(x=box_positive, shape=[1, -1, 8])
            score_reshape = fluid.layers.reshape(
                x=score_slice, shape=[1, 1, -1])
            pred_result = fluid.layers.multiclass_nms(
                bboxes=box_reshape,
                scores=score_reshape,
                score_threshold=cfg.TEST.score_thresh,
                nms_top_k=-1,
                nms_threshold=cfg.TEST.nms_thresh,
                keep_top_k=cfg.TEST.detections_per_im,
                normalized=False,
                background_label=-1)
            result_shape = fluid.layers.shape(pred_result)
            res_dimension = fluid.layers.slice(
                result_shape, axes=[0], starts=[1], ends=[2])
            res_dimension = fluid.layers.reshape(res_dimension, shape=[1, 1])
            dimension = fluid.layers.fill_constant(
                shape=[1, 1], value=2, dtype='int32')
            cond = fluid.layers.less_than(dimension, res_dimension)

            def case1():
                res = fluid.layers.create_global_var(
                    shape=[1, 10],
                    value=0.0,
                    dtype='float32',
                    persistable=False)
                coordinate = fluid.layers.fill_constant(
                    shape=[9], value=0.0, dtype='float32')
                pred_class = fluid.layers.fill_constant(
                    shape=[1], value=i + 1, dtype='float32')
                add_class = fluid.layers.concat(
                    [pred_class, coordinate], axis=0)
                normal_result = fluid.layers.elementwise_add(pred_result,
                                                             add_class)
                fluid.layers.assign(normal_result, res)
                return res

            def case2():
                res = fluid.layers.create_global_var(
                    shape=[1, 10],
                    value=0.0,
                    dtype='float32',
                    persistable=False)
                normal_result = fluid.layers.fill_constant(
                    shape=[1, 10], value=-1.0, dtype='float32')
                fluid.layers.assign(normal_result, res)
                return res

            res = fluid.layers.case(
                pred_fn_pairs=[(cond, case1)], default=case2)
            results.append(res)
        if len(results) == 1:
            self.pred_result = results[0]
            return
        outs = []
        out = fluid.layers.concat(results)
        zero = fluid.layers.fill_constant(
            shape=[1, 1], value=0.0, dtype='float32')
        out_split, _ = fluid.layers.split(out, dim=1, num_or_sections=[1, 9])
        out_bool = fluid.layers.greater_than(out_split, zero)
        idx = fluid.layers.where(out_bool)
        idx_split, _ = fluid.layers.split(idx, dim=1, num_or_sections=[1, 1])
        idx = fluid.layers.reshape(idx_split, [-1, 1])
        self.pred_result = fluid.layers.gather(input=out, index=idx)

    def rpn_heads(self, rpn_input):
        # RPN hidden representation
        dim_out = rpn_input.shape[1]
        rpn_conv = fluid.layers.conv2d(
            input=rpn_input,
            num_filters=dim_out,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            name='conv_rpn',
            param_attr=ParamAttr(
                name="conv_rpn_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="conv_rpn_b", learning_rate=2., regularizer=L2Decay(0.)))
        self.anchor, self.var = rotated_anchor_generator(
            input=rpn_conv,
            anchor_sizes=cfg.anchor_sizes,
            aspect_ratios=cfg.aspect_ratios,
            angles=cfg.anchor_angle,
            variance=cfg.variance,
            stride=cfg.rpn_stride,
            offset=0.5)
        num_anchor = self.anchor.shape[2]
        # Proposal classification scores
        self.rpn_cls_score = fluid.layers.conv2d(
            rpn_conv,
            num_filters=num_anchor,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            name='rpn_cls_score',
            param_attr=ParamAttr(
                name="rpn_cls_logits_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_cls_logits_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        # Proposal bbox regression deltas
        self.rpn_bbox_pred = fluid.layers.conv2d(
            rpn_conv,
            num_filters=5 * num_anchor,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            name='rpn_bbox_pred',
            param_attr=ParamAttr(
                name="rpn_bbox_pred_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_bbox_pred_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        rpn_cls_score_prob = fluid.layers.sigmoid(
            self.rpn_cls_score, name='rpn_cls_score_prob')

        param_obj = cfg.TRAIN if self.mode == 'train' else cfg.TEST
        pre_nms_top_n = param_obj.rpn_pre_nms_top_n
        post_nms_top_n = param_obj.rpn_post_nms_top_n
        nms_thresh = param_obj.rpn_nms_thresh
        min_size = param_obj.rpn_min_size
        self.rpn_rois, self.rpn_roi_probs = rotated_generate_proposals(
            scores=rpn_cls_score_prob,
            bbox_deltas=self.rpn_bbox_pred,
            im_info=self.im_info,
            anchors=self.anchor,
            variances=self.var,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=param_obj.rpn_nms_thresh,
            min_size=param_obj.rpn_min_size)
        if self.mode == 'train':
            outs = rotated_generate_proposal_labels(
                rpn_rois=self.rpn_rois,
                gt_classes=self.gt_label,
                is_crowd=self.is_crowd,
                gt_boxes=self.gt_box,
                im_info=self.im_info,
                batch_size_per_im=cfg.TRAIN.batch_size_per_im,
                fg_fraction=cfg.TRAIN.fg_fractrion,
                fg_thresh=cfg.TRAIN.fg_thresh,
                bg_thresh_hi=cfg.TRAIN.bg_thresh_hi,
                bg_thresh_lo=cfg.TRAIN.bg_thresh_lo,
                bbox_reg_weights=cfg.bbox_reg_weights,
                class_nums=cfg.class_num,
                use_random=self.use_random)

            self.rois = outs[0]
            self.labels_int32 = outs[1]
            self.bbox_targets = outs[2]
            self.bbox_inside_weights = outs[3]
            self.bbox_outside_weights = outs[4]

    def fast_rcnn_heads(self, roi_input):
        if self.mode == 'train':
            pool_rois = self.rois
        else:
            pool_rois = self.rpn_rois
        pool = rotated_roi_align(
            input=roi_input,
            rois=pool_rois,
            pooled_height=cfg.roi_resolution,
            pooled_width=cfg.roi_resolution,
            spatial_scale=cfg.spatial_scale)
        self.res5_2_sum = self.add_roi_box_head_func(pool)
        rcnn_out = fluid.layers.pool2d(
            self.res5_2_sum, pool_type='avg', pool_size=7, name='res5_pool')
        self.cls_score = fluid.layers.fc(input=rcnn_out,
                                         size=cfg.class_num,
                                         act=None,
                                         name='cls_score',
                                         param_attr=ParamAttr(
                                             name='cls_score_w',
                                             initializer=Normal(
                                                 loc=0.0, scale=0.001)),
                                         bias_attr=ParamAttr(
                                             name='cls_score_b',
                                             learning_rate=2.,
                                             regularizer=L2Decay(0.)))
        self.bbox_pred = fluid.layers.fc(input=rcnn_out,
                                         size=5 * cfg.class_num,
                                         act=None,
                                         name='bbox_pred',
                                         param_attr=ParamAttr(
                                             name='bbox_pred_w',
                                             initializer=Normal(
                                                 loc=0.0, scale=0.01)),
                                         bias_attr=ParamAttr(
                                             name='bbox_pred_b',
                                             learning_rate=2.,
                                             regularizer=L2Decay(0.)))

    def fast_rcnn_loss(self):
        labels_int64 = fluid.layers.cast(x=self.labels_int32, dtype='int64')
        labels_int64.stop_gradient = True
        loss_cls = fluid.layers.softmax_with_cross_entropy(
            logits=self.cls_score,
            label=labels_int64,
            numeric_stable_mode=True, )
        loss_cls = fluid.layers.reduce_mean(loss_cls)
        loss_bbox = fluid.layers.smooth_l1(
            x=self.bbox_pred,
            y=self.bbox_targets,
            inside_weight=self.bbox_inside_weights,
            outside_weight=self.bbox_outside_weights,
            sigma=1.0)
        loss_bbox = fluid.layers.reduce_mean(loss_bbox)
        return loss_cls, loss_bbox

    def rpn_loss(self):
        rpn_cls_score_reshape = fluid.layers.transpose(
            self.rpn_cls_score, perm=[0, 2, 3, 1])
        rpn_bbox_pred_reshape = fluid.layers.transpose(
            self.rpn_bbox_pred, perm=[0, 2, 3, 1])

        anchor_reshape = fluid.layers.reshape(self.anchor, shape=(-1, 5))
        var_reshape = fluid.layers.reshape(self.var, shape=(-1, 5))

        rpn_cls_score_reshape = fluid.layers.reshape(
            x=rpn_cls_score_reshape, shape=(0, -1, 1))
        rpn_bbox_pred_reshape = fluid.layers.reshape(
            x=rpn_bbox_pred_reshape, shape=(0, -1, 5))
        score_pred, loc_pred, score_tgt, loc_tgt = \
            rrpn_target_assign(
                bbox_pred=rpn_bbox_pred_reshape,
                cls_logits=rpn_cls_score_reshape,
                anchor_box=anchor_reshape,
                gt_boxes=self.gt_box,
                im_info=self.im_info,
                rpn_batch_size_per_im=cfg.TRAIN.rpn_batch_size_per_im,
                rpn_straddle_thresh=-1,
                rpn_fg_fraction=cfg.TRAIN.rpn_fg_fraction,
                rpn_positive_overlap=cfg.TRAIN.rpn_positive_overlap,
                rpn_negative_overlap=cfg.TRAIN.rpn_negative_overlap,
                use_random=self.use_random)
        score_tgt = fluid.layers.cast(x=score_tgt, dtype='float32')
        rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=score_pred, label=score_tgt)
        rpn_cls_loss = fluid.layers.reduce_mean(
            rpn_cls_loss, name='loss_rpn_cls')

        rpn_reg_loss = fluid.layers.smooth_l1(x=loc_pred, y=loc_tgt, sigma=3.0)
        rpn_reg_loss = fluid.layers.reduce_sum(
            rpn_reg_loss, name='loss_rpn_bbox')
        score_shape = fluid.layers.shape(score_tgt)
        score_shape = fluid.layers.cast(x=score_shape, dtype='float32')
        norm = fluid.layers.reduce_prod(score_shape)
        norm.stop_gradient = True
        rpn_reg_loss = rpn_reg_loss / norm
        return rpn_cls_loss, rpn_reg_loss
