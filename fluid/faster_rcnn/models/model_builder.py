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
from paddle.fluid.regularizer import L2Decay


class FasterRCNN(object):
    def __init__(self,
                 cfg=None,
                 add_conv_body_func=None,
                 add_roi_box_head_func=None,
                 is_train=True,
                 use_pyreader=True,
                 use_random=True):
        self.add_conv_body_func = add_conv_body_func
        self.add_roi_box_head_func = add_roi_box_head_func
        self.cfg = cfg
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

    def loss(self):
        # Fast RCNN loss
        loss_cls, loss_bbox = self.fast_rcnn_loss()
        # RPN loss
        rpn_cls_loss, rpn_reg_loss = self.rpn_loss()
        return loss_cls, loss_bbox, rpn_cls_loss, rpn_reg_loss,

    def eval_out(self):
        cls_prob = fluid.layers.softmax(self.cls_score, use_cudnn=False)
        return [self.rpn_rois, cls_prob, self.bbox_pred]

    def build_input(self, image_shape):
        if self.use_pyreader:
            self.py_reader = fluid.layers.py_reader(
                capacity=64,
                shapes=[[-1] + image_shape, [-1, 4], [-1, 1], [-1, 1], [-1, 3],
                        [-1, 1]],
                lod_levels=[0, 1, 1, 1, 0, 0],
                dtypes=[
                    "float32", "float32", "int32", "int32", "float32", "int32"
                ],
                use_double_buffer=True)
            self.image, self.gt_box, self.gt_label, self.is_crowd, \
                self.im_info, self.im_id = fluid.layers.read_file(self.py_reader)
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

    def feeds(self):
        if not self.is_train:
            return [self.image, self.im_info, self.im_id]
        return [
            self.image, self.gt_box, self.gt_label, self.is_crowd, self.im_info,
            self.im_id
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
            anchor_sizes=self.cfg.anchor_sizes,
            aspect_ratios=self.cfg.aspect_ratios,
            variance=self.cfg.variance,
            stride=[16.0, 16.0])
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

        pre_nms_top_n = 12000 if self.is_train else 6000
        post_nms_top_n = 2000 if self.is_train else 1000
        rpn_rois, rpn_roi_probs = fluid.layers.generate_proposals(
            scores=rpn_cls_score_prob,
            bbox_deltas=self.rpn_bbox_pred,
            im_info=self.im_info,
            anchors=self.anchor,
            variances=self.var,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=0.7,
            min_size=0.0,
            eta=1.0)
        self.rpn_rois = rpn_rois
        if self.is_train:
            outs = fluid.layers.generate_proposal_labels(
                rpn_rois=rpn_rois,
                gt_classes=self.gt_label,
                is_crowd=self.is_crowd,
                gt_boxes=self.gt_box,
                im_info=self.im_info,
                batch_size_per_im=self.cfg.batch_size_per_im,
                fg_fraction=0.25,
                fg_thresh=0.5,
                bg_thresh_hi=0.5,
                bg_thresh_lo=0.0,
                bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                class_nums=self.cfg.class_num,
                use_random=self.use_random)

            self.rois = outs[0]
            self.labels_int32 = outs[1]
            self.bbox_targets = outs[2]
            self.bbox_inside_weights = outs[3]
            self.bbox_outside_weights = outs[4]

    def fast_rcnn_heads(self, roi_input):
        if self.is_train:
            pool_rois = self.rois
        else:
            pool_rois = self.rpn_rois
        pool = fluid.layers.roi_pool(
            input=roi_input,
            rois=pool_rois,
            pooled_height=14,
            pooled_width=14,
            spatial_scale=0.0625)
        rcnn_out = self.add_roi_box_head_func(pool)
        self.cls_score = fluid.layers.fc(input=rcnn_out,
                                         size=self.cfg.class_num,
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
                                         size=4 * self.cfg.class_num,
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
        #loss_cls = fluid.layers.softmax_with_cross_entropy(
        #        logits=cls_score,
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

        score_pred, loc_pred, score_tgt, loc_tgt = \
            fluid.layers.rpn_target_assign(
                bbox_pred=rpn_bbox_pred_reshape,
                cls_logits=rpn_cls_score_reshape,
                anchor_box=anchor_reshape,
                anchor_var=var_reshape,
                gt_boxes=self.gt_box,
                is_crowd=self.is_crowd,
                im_info=self.im_info,
                rpn_batch_size_per_im=256,
                rpn_straddle_thresh=0.0,
                rpn_fg_fraction=0.5,
                rpn_positive_overlap=0.7,
                rpn_negative_overlap=0.3,
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
