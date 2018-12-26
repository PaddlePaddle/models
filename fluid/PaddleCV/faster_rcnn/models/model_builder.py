#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import cPickle as cp
import numpy as np


class FasterRCNN(object):
    def __init__(self,
                 add_conv_body_func=None,
                 add_roi_box_head_func=None,
                 is_train=True,
                 use_pyreader=True,
                 use_random=True):
        self.add_conv_body_func = add_conv_body_func
        self.add_roi_box_head_func = add_roi_box_head_func
        self.is_train = is_train
        self.use_pyreader = use_pyreader
        self.use_random = use_random
        #self.py_reader = None

    def build_model(self, image_shape):
        self.build_input(image_shape)
        body_conv = self.add_conv_body_func(self.image)
        # RPN
        self.rpn_heads(body_conv)
        # Fast RCNN
        self.fast_rcnn_heads(body_conv)
        # Mask RCNN
        if cfg.MASK_ON:
            self.mask_rcnn_heads(body_conv)

    def loss(self):
        # Fast RCNN loss
        loss_cls, loss_bbox = self.fast_rcnn_loss()
        # RPN loss
        rpn_cls_loss, rpn_reg_loss = self.rpn_loss()
        if cfg.MASK_ON:
            loss_mask = self.mask_rcnn_loss()
            return loss_cls, loss_bbox, rpn_cls_loss, rpn_reg_loss, loss_mask
        return loss_cls, loss_bbox, rpn_cls_loss, rpn_reg_loss,

    def eval_bbox_out(self):
        cls_prob = fluid.layers.softmax(self.cls_score, use_cudnn=False)
        return [self.rpn_rois, cls_prob, self.bbox_pred]

    def eval_mask_out(self):
        return self.mask_fcn_logits

    def eval(self):
        return self.pred_result

    def build_input(self, image_shape):
        if self.use_pyreader:
            if cfg.MASK_ON:
                self.py_reader = fluid.layers.py_reader(
                    capacity=64,
                    shapes=[[-1] + image_shape, [-1, 4], [-1, 1], [-1, 1],
                        [-1, 3], [-1, 1], [-1] + image_shape[1:]],
                    lod_levels=[0, 1, 1, 1, 0, 0, 1],
                    dtypes=[
                        "float32", "float32", "int32", "int32",\
                        "float32", "int32", "uint8"
                    ],
                    use_double_buffer=True)
                self.image, self.gt_box, self.gt_label, self.is_crowd, \
                self.im_info, self.im_id, self.gt_masks = fluid.layers.read_file(
                                                              self.py_reader)
            else:
                self.py_reader = fluid.layers.py_reader(
                    capacity=64,
                    shapes=[[-1] + image_shape, [-1, 4], [-1, 1], [-1, 1],
                        [-1, 3], [-1, 1]],
                    lod_levels=[0, 1, 1, 1, 0, 0],
                    dtypes=[
                        "float32", "float32", "int32", "int32",\
                        "float32", "int32"
                    ],
                    use_double_buffer=True)
                self.image, self.gt_box, self.gt_label, self.is_crowd, \
                self.im_info, self.im_id = fluid.layers.read_file(
                                                    self.py_reader)
        else:
            self.image = fluid.layers.data(
                name='image', shape=image_shape, dtype='float32')
            self.gt_box = fluid.layers.data(
                name='gt_box', shape=[4], dtype='float32', lod_level=1)
            self.gt_label = fluid.layers.data(
                name='gt_label', shape=[1], dtype='int32', lod_level=1)
            self.is_crowd = fluid.layers.data(
                name='is_crowd',
                shape=[-1],
                dtype='int32',
                lod_level=1,
                append_batch_size=False)
            self.im_info = fluid.layers.data(
                name='im_info', shape=[3], dtype='float32')
            self.im_id = fluid.layers.data(
                name='im_id', shape=[1], dtype='int32')
            if cfg.MASK_ON:
                self.gt_masks = fluid.layers.data(
                    name='gt_masks',
                    shape=image_shape[1:],
                    dtype='uint8',
                    lod_level=1)

    def feeds(self):
        if not self.is_train:
            return [self.image, self.im_info, self.im_id]
        if not cfg.MASK_ON:
            return [
                self.image, self.gt_box, self.gt_label, self.is_crowd,
                self.im_info, self.im_id
            ]
        return [
            self.image, self.gt_box, self.gt_label, self.is_crowd, self.im_info,
            self.im_id, self.gt_masks
        ]

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
        self.anchor, self.var = fluid.layers.anchor_generator(
            input=rpn_conv,
            anchor_sizes=cfg.anchor_sizes,
            aspect_ratios=cfg.aspect_ratio,
            variance=cfg.variances,
            stride=cfg.rpn_stride)
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
            num_filters=4 * num_anchor,
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

        param_obj = cfg.TRAIN if self.is_train else cfg.TEST
        pre_nms_top_n = param_obj.rpn_pre_nms_top_n
        post_nms_top_n = param_obj.rpn_post_nms_top_n
        nms_thresh = param_obj.rpn_nms_thresh
        min_size = param_obj.rpn_min_size
        eta = param_obj.rpn_eta
        self.rpn_rois, self.rpn_roi_probs = fluid.layers.generate_proposals(
            scores=rpn_cls_score_prob,
            bbox_deltas=self.rpn_bbox_pred,
            im_info=self.im_info,
            anchors=self.anchor,
            variances=self.var,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=nms_thresh,
            min_size=min_size,
            eta=eta)
        if self.is_train:
            outs = fluid.layers.generate_proposal_labels(
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

            if cfg.MASK_ON:
                mask_out = fluid.layers.generate_mask_labels(
                    im_info=self.im_info,
                    gt_classes=self.gt_label,
                    is_crowd=self.is_crowd,
                    gt_segms=self.gt_masks,
                    rois=self.rois,
                    labels_int32=self.labels_int32,
                    num_classes=cfg.class_num,
                    resolution=cfg.resolution)
                self.mask_rois = mask_out[0]
                self.roi_has_mask_int32 = mask_out[1]
                self.mask_int32 = mask_out[2]

                #self.roi_has_mask_int32.persistable = True
                #print(self.roi_has_mask_int32)
                #self.mask_int32.persistable = True

    def fast_rcnn_heads(self, roi_input):
        if self.is_train:
            pool_rois = self.rois
        else:
            pool_rois = self.rpn_rois
        self.res5_2_sum = self.add_roi_box_head_func(roi_input, pool_rois)
        self.rcnn_out = fluid.layers.pool2d(
            self.res5_2_sum, pool_type='avg', pool_size=7, name='res5_pool')
        self.cls_score = fluid.layers.fc(input=self.rcnn_out,
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
        self.bbox_pred = fluid.layers.fc(input=self.rcnn_out,
                                         size=4 * cfg.class_num,
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

    def SuffixNet(self, conv5):
        mask_out = fluid.layers.conv2d_transpose(
            input=conv5,
            num_filters=cfg.DIM_REDUCED,
            filter_size=2,
            stride=2,
            act='relu',
            param_attr=ParamAttr(
                name='conv5_mask_w', initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name='conv5_mask_b', learning_rate=2., regularizer=L2Decay(0.)))
        act_func = None
        if not self.is_train:
            act_func = 'sigmoid'
        mask_fcn_logits = fluid.layers.conv2d(
            input=mask_out,
            num_filters=cfg.class_num,
            filter_size=1,
            act=act_func,
            param_attr=ParamAttr(
                name='mask_fcn_logits_w', initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name="mask_fcn_logits_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        if self.is_train:
            mask_fcn_logits = fluid.layers.lod_reset(mask_fcn_logits,
                                                     self.mask_int32)
        else:
            mask_fcn_logits = fluid.layers.lod_reset(mask_fcn_logits,
                                                     self.pred_result)
        return mask_fcn_logits

    def mask_rcnn_heads(self, mask_input):
        if self.is_train:
            conv5 = fluid.layers.gather(self.res5_2_sum,
                                        self.roi_has_mask_int32)
            self.mask_fcn_logits = self.SuffixNet(conv5)
        else:
            im_scale = fluid.layers.slice(
                self.im_info, [1], starts=[2], ends=[3])
            im_scale_lod = fluid.layers.sequence_expand(im_scale, self.rpn_rois)
            boxes = self.rpn_rois / im_scale_lod
            cls_prob = fluid.layers.softmax(self.cls_score, use_cudnn=False)
            bbox_pred_reshape = fluid.layers.reshape(self.bbox_pred,
                                                     (-1, cfg.class_num, 4))
            bbox_pred_trans = fluid.layers.transpose(
                bbox_pred_reshape, perm=[1, 0, 2])
            bbox_var = fluid.layers.create_tensor(dtype='float32')
            fluid.layers.assign(
                np.array(
                    cfg.bbox_reg_weights, dtype='float32'), bbox_var)
            self.pred_result = fluid.layers.detection_output(
                loc=bbox_pred_trans,
                scores=cls_prob,
                prior_box=boxes,
                prior_box_var=bbox_var,
                im_info=self.im_info,
                nms_threshold=cfg.TEST.nms_thresh,
                nms_top_k=-1,
                keep_top_k=cfg.TEST.detections_per_im,
                score_threshold=cfg.TEST.score_thresh,
                shared=False)
            pred_res_shape = fluid.layers.shape(self.pred_result)
            shape = fluid.layers.reduce_prod(pred_res_shape)
            shape = fluid.layers.reshape(shape, [1, 1])
            ones = fluid.layers.fill_constant([1, 1], value=1, dtype='int32')
            cond = fluid.layers.equal(x=shape, y=ones)
            ie = fluid.layers.IfElse(cond)

            with ie.true_block():
                pred_res_null = ie.input(self.pred_result)
                ie.output(pred_res_null)
            with ie.false_block():
                pred_res = ie.input(self.pred_result)
                pred_boxes = fluid.layers.slice(
                    pred_res, [1], starts=[2], ends=[6])
                pred_boxes = fluid.layers.lod_reset(pred_boxes,
                                                    self.pred_result)
                im_scale_lod = fluid.layers.sequence_expand(im_scale,
                                                            pred_boxes)
                mask_rois = pred_boxes * im_scale_lod
                conv5 = self.add_roi_box_head_func(mask_input, mask_rois)
                mask_fcn = self.SuffixNet(conv5)
                ie.output(mask_fcn)
            self.mask_fcn_logits = ie()[0]
            self.mask_fcn_logits.persistable = True

    def mask_rcnn_loss(self):
        mask_label = fluid.layers.cast(x=self.mask_int32, dtype='float32')
        reshape_dim = cfg.class_num * cfg.resolution * cfg.resolution
        self.mask_fcn_logits_reshape = fluid.layers.reshape(
            self.mask_fcn_logits, (-1, reshape_dim))
        loss_mask = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=self.mask_fcn_logits_reshape, label=mask_label, ignore_index=-1)

        mask_label_clip = fluid.layers.clip(mask_label, -1., 0.)
        sums = fluid.layers.reduce_sum(mask_label_clip)
        mask_shape = fluid.layers.shape(mask_label_clip)
        mask_shape = fluid.layers.cast(x=mask_shape, dtype='float32')
        mask_num = fluid.layers.reduce_prod(mask_shape)
        norm = mask_num + sums
        norm.stop_gradient = True

        loss_mask = fluid.layers.reduce_sum(loss_mask, name='loss_mask')
        loss_mask = loss_mask / norm

        return loss_mask

    def fast_rcnn_loss(self):
        labels_int64 = fluid.layers.cast(x=self.labels_int32, dtype='int64')
        labels_int64.stop_gradient = True
        #loss_cls = fluid.layers.softmax_with_cross_entropy(
        #        logits=self.cls_score,
        #        label=labels_int64
        #        )
        cls_prob = fluid.layers.softmax(self.cls_score, use_cudnn=False)
        loss_cls = fluid.layers.cross_entropy(cls_prob, labels_int64)
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

        anchor_reshape = fluid.layers.reshape(self.anchor, shape=(-1, 4))
        var_reshape = fluid.layers.reshape(self.var, shape=(-1, 4))

        rpn_cls_score_reshape = fluid.layers.reshape(
            x=rpn_cls_score_reshape, shape=(0, -1, 1))
        rpn_bbox_pred_reshape = fluid.layers.reshape(
            x=rpn_bbox_pred_reshape, shape=(0, -1, 4))
        score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight = \
            fluid.layers.rpn_target_assign(
                bbox_pred=rpn_bbox_pred_reshape,
                cls_logits=rpn_cls_score_reshape,
                anchor_box=anchor_reshape,
                anchor_var=var_reshape,
                gt_boxes=self.gt_box,
                is_crowd=self.is_crowd,
                im_info=self.im_info,
                rpn_batch_size_per_im=cfg.TRAIN.rpn_batch_size_per_im,
                rpn_straddle_thresh=cfg.TRAIN.rpn_straddle_thresh,
                rpn_fg_fraction=cfg.TRAIN.rpn_fg_fraction,
                rpn_positive_overlap=cfg.TRAIN.rpn_positive_overlap,
                rpn_negative_overlap=cfg.TRAIN.rpn_negative_overlap,
                use_random=self.use_random)
        score_tgt = fluid.layers.cast(x=score_tgt, dtype='float32')
        rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=score_pred, label=score_tgt)
        rpn_cls_loss = fluid.layers.reduce_mean(
            rpn_cls_loss, name='loss_rpn_cls')

        rpn_reg_loss = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=3.0,
            inside_weight=bbox_weight,
            outside_weight=bbox_weight)
        rpn_reg_loss = fluid.layers.reduce_sum(
            rpn_reg_loss, name='loss_rpn_bbox')
        score_shape = fluid.layers.shape(score_tgt)
        score_shape = fluid.layers.cast(x=score_shape, dtype='float32')
        norm = fluid.layers.reduce_prod(score_shape)
        norm.stop_gradient = True
        rpn_reg_loss = rpn_reg_loss / norm
        return rpn_cls_loss, rpn_reg_loss
