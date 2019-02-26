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
import FPN


class RCNN(object):
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
        self.spatial_scale = cfg.spatial_scale

    def build_model(self, image_shape):
        self.build_input(image_shape)
        # backbone
        body_dict, body_name_list = self.add_conv_body_func(self.image)
        if cfg.FPN_ON:
            body_dict, self.spatial_scale, body_name_list = FPN.add_fpn_onto_conv_body(
                body_dict)
        # RPN
        self.rpn_heads(body_dict, body_name_list)
        # Fast RCNN
        self.fast_rcnn_heads(body_dict, body_name_list)
        if not self.is_train:
            self.eval_bbox()
        # Mask RCNN
        if cfg.MASK_ON:
            self.mask_rcnn_heads(body_dict, body_name_list)

    def loss(self):
        losses = []
        rkeys = ['loss']
        # Fast RCNN loss
        loss_cls, loss_bbox = self.fast_rcnn_loss()
        losses = losses + [loss_cls, loss_bbox]
        rkeys = rkeys + ['loss_cls', 'loss_bbox']
        # RPN loss
        if cfg.FPN_ON:
            loss_rpn_cls_name, loss_rpn_bbox_name, loss_rpn_cls_list, loss_rpn_bbox_list = self.fpn_rpn_loss(
            )
            losses = losses + loss_rpn_cls_list + loss_rpn_bbox_list
            rkeys = rkeys + loss_rpn_cls_name + loss_rpn_bbox_name
        else:
            rpn_cls_loss, rpn_reg_loss = self.single_scale_rpn_loss()
            losses = losses + [rpn_cls_loss, rpn_reg_loss]
            rkeys = rkeys + [
                'loss_rpn_cls',
                'loss_rpn_bbox',
            ]
        if cfg.MASK_ON:
            loss_mask = self.mask_rcnn_loss()
            losses = losses + [loss_mask]
            rkeys = rkeys + ["loss_mask"]
        loss = fluid.layers.sum(losses)
        rloss = [loss] + losses
        return rloss, rkeys

    def eval_mask_out(self):
        return self.mask_fcn_logits

    def eval_bbox_out(self):
        return self.pred_result

    def build_input(self, image_shape):
        if self.use_pyreader:
            in_shapes = [[-1] + image_shape, [-1, 4], [-1, 1], [-1, 1],
                         [-1, 3], [-1, 1]]
            lod_levels = [0, 1, 1, 1, 0, 0]
            dtypes = [
                'float32', 'float32', 'int32', 'int32', 'float32', 'int32'
            ]
            if cfg.MASK_ON:
                in_shapes.append([-1, 2])
                lod_levels.append(3)
                dtypes.append('float32')
            self.py_reader = fluid.layers.py_reader(
                capacity=64,
                shapes=in_shapes,
                lod_levels=lod_levels,
                dtypes=dtypes,
                use_double_buffer=True)
            ins = fluid.layers.read_file(self.py_reader)
            self.image = ins[0]
            self.gt_box = ins[1]
            self.gt_label = ins[2]
            self.is_crowd = ins[3]
            self.im_info = ins[4]
            self.im_id = ins[5]
            if cfg.MASK_ON:
                self.gt_masks = ins[6]
        else:
            self.image = fluid.layers.data(
                name='image', shape=image_shape, dtype='float32')
            self.gt_box = fluid.layers.data(
                name='gt_box', shape=[4], dtype='float32', lod_level=1)
            self.gt_label = fluid.layers.data(
                name='gt_label', shape=[1], dtype='int32', lod_level=1)
            self.is_crowd = fluid.layers.data(
                name='is_crowd', shape=[1], dtype='int32', lod_level=1)
            self.im_info = fluid.layers.data(
                name='im_info', shape=[3], dtype='float32')
            self.im_id = fluid.layers.data(
                name='im_id', shape=[1], dtype='int32')
            if cfg.MASK_ON:
                self.gt_masks = fluid.layers.data(
                    name='gt_masks', shape=[2], dtype='float32', lod_level=3)

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

    def eval_bbox(self):
        self.im_scale = fluid.layers.slice(
            self.im_info, [1], starts=[2], ends=[3])
        im_scale_lod = fluid.layers.sequence_expand(self.im_scale,
                                                    self.rpn_rois)
        boxes = self.rpn_rois / im_scale_lod
        cls_prob = fluid.layers.softmax(self.cls_score, use_cudnn=False)
        bbox_pred_reshape = fluid.layers.reshape(self.bbox_pred,
                                                 (-1, cfg.class_num, 4))
        decoded_box = fluid.layers.box_coder(
            prior_box=boxes,
            prior_box_var=cfg.bbox_reg_weights,
            target_box=bbox_pred_reshape,
            code_type='decode_center_size',
            box_normalized=False,
            axis=1)
        cliped_box = fluid.layers.box_clip(
            input=decoded_box, im_info=self.im_info)
        self.pred_result = fluid.layers.multiclass_nms(
            bboxes=cliped_box,
            scores=cls_prob,
            score_threshold=cfg.TEST.score_thresh,
            nms_top_k=-1,
            nms_threshold=cfg.TEST.nms_thresh,
            keep_top_k=cfg.TEST.detections_per_im,
            normalized=False)

    def rpn_heads(self, body_dict, body_name_list):
        if cfg.FPN_ON:
            self.rois = self.FPN_rpn_heads(body_dict, body_name_list)
        else:
            self.rois = self.single_scale_rpn_heads(body_dict, body_name_list)

    def FPN_rpn_heads(self, fpn_dict, fpn_name_list):
        fpn_outputs = FPN.add_fpn_rpn_outputs(fpn_dict, self.im_info,
                                              fpn_name_list)
        self.rpn_fpn_list = fpn_outputs[0]
        self.rpn_rois_list = fpn_outputs[1]
        self.rpn_roi_probs_list = fpn_outputs[2]
        self.anchors_list = fpn_outputs[3]
        self.var_list = fpn_outputs[4]
        rois_collect = fluid.layers.collect_fpn_proposals(
            self.rpn_rois_list, self.rpn_roi_probs_list)
        if self.is_train:
            rois_collect = self.generate_labels(rois_collect)
        rois, self.restore_index = fluid.layers.distribute_fpn_proposals(
            rois_collect)
        return rois
        # TODO: 
        # collect
        # generate_proposal_labels
        # distribute

    def single_scale_rpn_heads(self, res_dict, res_name_list):
        rpn_input_name = res_name_list[-1]
        rpn_input = res_dict[rpn_input_name]
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
        rois, self.rpn_roi_probs = fluid.layers.generate_proposals(
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
            rois = self.generate_labels(rois)
        return rois

    def generate_labels(self, input_rois):
        outs = fluid.layers.generate_proposal_labels(
            rpn_rois=input_rois,
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

        rois = outs[0]
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
                rois=rois,
                labels_int32=self.labels_int32,
                num_classes=cfg.class_num,
                resolution=cfg.resolution)
            self.mask_rois = mask_out[0]
            self.roi_has_mask_int32 = mask_out[1]
            self.mask_int32 = mask_out[2]
        return rois

    def add_FPN_roi_head_output(body_dict, pool_rois, body_name_list,
                                spatial_scale):
        roi_out_list = FPN.add_FPN_roi_head(body_dict, pool_rois,
                                            body_name_list, spatial_scale)
        roi_feat_shuffle = fluid.layers.concat(roi_out_list)
        roi_feat = fluid.layers.gather(roi_feat_shuffle, self.restore_index)
        roi_feat = fluid.layers.lod_reset(self.roi_feat, pool_rois)
        fc6 = fluid.layers.fc(input=roi_feat,
                              size=cfg.MLP_HEAD_DIM,
                              act='relu',
                              name='fc6',
                              param_attr=ParamAttr(name='fc6_w'),
                              bias_attr=ParamAttr(
                                  name='fc6_b',
                                  learning_rate=2.,
                                  regularizer=L2Decay(0.)))
        fc7 = fluid.layers.fc(input=fc6,
                              size=cfg.MLP_HEAD_DIM,
                              act='relu',
                              name='fc7',
                              param_attr=ParamAttr(name='fc7_w'),
                              bias_attr=ParamAttr(
                                  name='fc7_b',
                                  learning_rate=2.,
                                  regularizer=L2Decay(0.)))
        return roi_feat, fc7

    def fast_rcnn_heads(self, body_dict, body_name_list):
        pool_rois = self.rois
        self.roi_feat, rcnn_out = self.add_roi_box_head_func(
            body_dict, pool_rois, body_name_list, self.spatial_scale)
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
            num_filters=cfg.dim_reduced,
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

        if not self.is_train:
            mask_fcn_logits = fluid.layers.lod_reset(mask_fcn_logits,
                                                     self.pred_result)
        return mask_fcn_logits

    def mask_rcnn_heads(self, body_dict, body_name_list):
        if self.is_train:
            conv5 = fluid.layers.gather(self.roi_feat, self.roi_has_mask_int32)
            self.mask_fcn_logits = self.SuffixNet(conv5)
        else:
            self.eval_bbox()
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
                im_scale_lod = fluid.layers.sequence_expand(self.im_scale,
                                                            pred_boxes)
                mask_rois = pred_boxes * im_scale_lod
                conv5 = self.add_roi_box_head_func(
                    body_dict, mask_rois, body_name_list, self.spatial_scale)
                mask_fcn = self.SuffixNet(conv5)
                ie.output(mask_fcn)
            self.mask_fcn_logits = ie()[0]

    def mask_rcnn_loss(self):
        mask_label = fluid.layers.cast(x=self.mask_int32, dtype='float32')
        reshape_dim = cfg.class_num * cfg.resolution * cfg.resolution
        mask_fcn_logits_reshape = fluid.layers.reshape(self.mask_fcn_logits,
                                                       (-1, reshape_dim))

        loss_mask = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=mask_fcn_logits_reshape,
            label=mask_label,
            ignore_index=-1,
            normalize=True)
        loss_mask = fluid.layers.reduce_sum(loss_mask, name='loss_mask')
        return loss_mask

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

    def single_scale_rpn_loss(self):
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
        if cfg.FPN_ON:
            rpn_cls_loss = fluid.layers.reduce_sum(rpn_cls_loss)
            rpn_cls_loss = rpn_cls_loss / (cfg.im_per_batch *
                                           cfg.TRAIN.rpn_batch_size_per_im)
        else:
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

    def fpn_rpn_loss(self):
        k_max = cfg.FPN_rpn_max_level
        k_min = cfg.FPN_rpn_min_level
        loss_rpn_cls_fpn_name = []
        loss_rpn_bbox_fpn_name = []
        loss_rpn_cls_fpn_list = []
        loss_rpn_bbox_fpn_list = []
        for lvl in range(k_min, k_max + 1):
            slvl = str(lvl)
            self.rpn_cls_score = self.rpn_fpn_list[lvl - k_min][0]
            self.rpn_bbox_pred = self.rpn_fpn_list[lvl - k_min][1]
            self.anchor = self.anchors_list[lvl - k_min]
            self.var = self.var_list[lvl - k_min]
            loss_rpn_cls_fpn, loss_rpn_bbox_fpn = self.single_scale_rpn_loss()
            loss_rpn_cls_fpn_name.append('loss_rpn_cls_fpn' + slvl)
            loss_rpn_bbox_fpn_name.append('loss_rpn_bbox_fpn' + slvl)
            loss_rpn_cls_fpn_list.append(loss_rpn_cls_fpn)
            loss_rpn_bbox_fpn_list.append(loss_rpn_bbox_fpn)
        return loss_rpn_cls_fpn_name, loss_rpn_bbox_fpn_name, loss_rpn_cls_fpn_list, loss_rpn_bbox_fpn_list
