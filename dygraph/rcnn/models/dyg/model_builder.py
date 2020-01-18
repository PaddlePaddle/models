#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Normal
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.nn import Conv2D, Pool2D

from config import cfg
from .resnet import add_ResNet50_conv_body, bottleneck_list
from .pyfuncs import rpn_target_assign_pyfn, generate_proposal_labels_pyfn, generate_mask_labels_pyfn


class rpn_heads(fluid.dygraph.Layer):
    def __init__(self, name_scope, dim_out, num_anchor):
        super(rpn_heads, self).__init__()
        self.rpn_conv = fluid.dygraph.Conv2D(
            num_channels=1024,
            num_filters=1024,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            param_attr=ParamAttr(
                "conv_rpn_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                "conv_rpn_b", learning_rate=2., regularizer=L2Decay(0.)))

        # Proposal classification scores
        self.rpn_cls_score = fluid.dygraph.Conv2D(
            num_channels=1024,
            num_filters=1 * num_anchor,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(
                name="rpn_cls_logits_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_cls_logits_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))

        # Proposal bbox regression deltas
        self.rpn_bbox_pred = fluid.dygraph.Conv2D(
            num_channels=1024,
            num_filters=4 * num_anchor,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(
                name="rpn_bbox_pred_w", initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name="rpn_bbox_pred_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))

    def forward(self, inputs):
        x = self.rpn_conv(inputs)
        cs = self.rpn_cls_score(x)
        bp = self.rpn_bbox_pred(x)
        if cfg.enable_ce:
            print("rpn cls score: ", np.abs(cs.numpy()).mean(), cs.shape)
            print("rpn bbox pred: ", np.abs(bp.numpy()).mean(), bp.shape)
        return tuple((x, cs, bp))


class fast_rcnn_heads(fluid.dygraph.Layer):
    def __init__(self, name_scope, res5):
        super(fast_rcnn_heads, self).__init__()
        self.res5 = res5
        self.res5_pool = fluid.dygraph.Pool2D(pool_type='avg', pool_size=7)

        self.cls_score = fluid.dygraph.Linear(
            input_dim=2048,
            output_dim=cfg.class_num,
            act=None,
            param_attr=ParamAttr(
                name='cls_score_w', initializer=Normal(
                    loc=0.0, scale=0.001)),
            bias_attr=ParamAttr(
                name='cls_score_b', learning_rate=2., regularizer=L2Decay(0.)))
        self.bbox_pred = fluid.dygraph.Linear(
            input_dim=2048,
            output_dim=4 * cfg.class_num,
            act=None,
            param_attr=ParamAttr(
                name='bbox_pred_w', initializer=Normal(
                    loc=0.0, scale=0.01)),
            bias_attr=ParamAttr(
                name='bbox_pred_b', learning_rate=2., regularizer=L2Decay(0.)))

    def forward(self, inputs):
        res5_o = self.res5(inputs)
        x = self.res5_pool(res5_o)
        x = fluid.layers.squeeze(x, axes=[2, 3])
        cs = self.cls_score(x)
        bp = self.bbox_pred(x)
        if cfg.enable_ce:
            print("rcnn inputs: ", inputs.shape, res5_o.shape, x.shape)
            print("cls score: ", np.abs(cs.numpy()).mean(), cs.shape)
            print("bbox pred", np.abs(bp.numpy()).mean(), bp.shape)
        return tuple((res5_o, cs, bp))


class suffixnet(fluid.dygraph.Layer):
    def __init__(self, name_scope, cfg, mode='train'):
        super(suffixnet, self).__init__()
        self.cfg = cfg
        self.mode = mode

        self.mask_out = fluid.dygraph.Conv2DTranspose(
            num_channels=2048,
            num_filters=self.cfg.dim_reduced,
            filter_size=2,
            stride=2,
            act='relu',
            param_attr=ParamAttr(
                name='conv5_mask_w', initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name='conv5_mask_b', learning_rate=2., regularizer=L2Decay(0.)))
        self.mask_fcn_logits = fluid.dygraph.Conv2D(
            num_channels=self.cfg.dim_reduced,
            num_filters=self.cfg.class_num,
            filter_size=1,
            act='sigmoid' if self.mode != 'train' else None,
            param_attr=ParamAttr(
                name='mask_fcn_logits_w', initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name='mask_fcn_logits_b',
                learning_rate=2.,
                regularizer=L2Decay(0.0)))

    def forward(self, inputs):
        m_out = self.mask_out(inputs)
        m_f_logits = self.mask_fcn_logits(m_out)
        if cfg.enable_ce:
            print("mask inputs: ", inputs.shape)
            print("mask head 1: ", np.abs(m_out.numpy()).mean(), m_out.shape)
            print("mask head 2: ", np.abs(m_f_logits.numpy()).mean(),
                  m_f_logits.shape)

        return m_f_logits


class RCNN(fluid.dygraph.Layer):
    def __init__(self, name_scope, mode='train', cfg=None, use_random=True):
        super(RCNN, self).__init__()
        self.mode = mode
        self.cfg = cfg
        self.use_random = use_random

        self.add_conv_body_func = add_ResNet50_conv_body("resnet50")
        self.rpn_heads = rpn_heads("rpn_heads", dim_out=1024, num_anchor=15)
        self.res5 = bottleneck_list(
            "res5", ch_in=1024, ch_out=512, count=3, stride=2)
        self.fast_rcnn_heads = fast_rcnn_heads("fast_rcnn_heads", self.res5)
        self.suffixnet = suffixnet("suffixnet", cfg, mode)

    def forward(self,
                image,
                im_info,
                gt_box=None,
                gt_label=None,
                is_crowd=None,
                gt_masks=None):
        image = to_variable(image)
        self.im_info = to_variable(im_info)
        if self.mode == 'train':
            self.gt_box = to_variable(gt_box)
            self.gt_label = to_variable(gt_label)
            self.is_crowd = to_variable(is_crowd)
            if self.cfg.MASK_ON:
                self.gt_masks = to_variable(gt_masks)
        # Backbone
        backbones = self.add_conv_body_func(image)
        self.body_conv = backbones[-1]
        if cfg.enable_ce:
            self.res3 = backbones[-2]
            self.res2 = backbones[-3]

        # RPN
        self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_pred = self.rpn_heads(
            self.body_conv)

        # ROI
        self.generate_anchor()
        self.roi()
        if self.mode == 'train':
            rois = self.rois
            rois_lod_np = self.rois_lod.numpy()
        else:
            # infer only use one image
            rois = self.rpn_rois
            rois_lod_np = self.rpn_rois_lod.numpy()

        cur_l = 0
        new_lod = [cur_l]
        for l in rois_lod_np:
            cur_l += l
            new_lod.append(cur_l)

        lod_t = to_variable(np.asarray(new_lod))
        roi_feat = self.roi_extractor(self.body_conv, rois, lod_t)

        # Fast RCNN
        self.res5_2_sum, self.cls_score, self.bbox_pred = self.fast_rcnn_heads(
            roi_feat)

        # Mask RCNN
        if self.cfg.MASK_ON and self.mode == 'train':
            conv5 = fluid.layers.gather(self.res5_2_sum,
                                        self.roi_has_mask_int32)
            self.mask_fcn_logits = self.suffixnet(conv5)

        if self.mode == 'train':
            return self.loss()
        elif self.mode == 'eval':
            return self.eval_model_python()

    def generate_anchor(self, ):
        self.anchor, self.var = fluid.layers.anchor_generator(
            input=self.rpn_conv,
            anchor_sizes=cfg.anchor_sizes,
            aspect_ratios=cfg.aspect_ratio,
            variance=cfg.variances,
            stride=cfg.rpn_stride)
        if cfg.enable_ce:
            print("anchor: ", self.anchor.shape, self.var.shape)

    def roi(self, ):
        param_obj = cfg.TRAIN if self.mode == 'train' else cfg.TEST
        pre_nms_top_n = param_obj.rpn_pre_nms_top_n
        post_nms_top_n = param_obj.rpn_post_nms_top_n
        nms_thresh = param_obj.rpn_nms_thresh
        min_size = param_obj.rpn_min_size
        eta = param_obj.rpn_eta
        rpn_cls_score_prob = fluid.layers.sigmoid(
            self.rpn_cls_score, name='rpn_cls_score_prob')
        self.rpn_rois, self.rpn_roi_probs, self.rpn_rois_lod = fluid.layers.generate_proposals(
            scores=rpn_cls_score_prob,
            bbox_deltas=self.rpn_bbox_pred,
            anchors=self.anchor,
            variances=self.var,
            im_info=self.im_info,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=nms_thresh,
            min_size=min_size,
            eta=eta)
        if cfg.enable_ce:
            print("rpn rois: ", np.abs(self.rpn_rois.numpy()).mean(),
                  self.rpn_rois.shape)

        if self.mode == 'train':
            outs = generate_proposal_labels_pyfn(
                rpn_rois=self.rpn_rois,
                rpn_rois_lod=self.rpn_rois_lod,
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
            self.rois_lod = outs[5]
            if self.cfg.MASK_ON:
                mask_out = generate_mask_labels_pyfn(
                    im_info=self.im_info,
                    gt_classes=self.gt_label,
                    is_crowd=self.is_crowd,
                    gt_segms=self.gt_masks,
                    rois=self.rois,
                    rois_lod=self.rois_lod,
                    labels_int32=self.labels_int32,
                    num_classes=cfg.class_num,
                    resolution=cfg.resolution)
                self.mask_rois = mask_out[0]
                self.roi_has_mask_int32 = mask_out[1]
                self.mask_int32 = mask_out[2]

    def roi_extractor(self, base_feat, rois, lod_t):
        if cfg.roi_func == 'RoIPool':
            roi_feat = fluid.layers.roi_pool(
                base_feat,
                rois=rois,
                rois_lod=lod_t,
                pooled_height=cfg.roi_resolution,
                pooled_width=cfg.roi_resolution,
                spatial_scale=cfg.spatial_scale)

        elif cfg.roi_func == 'RoIAlign':
            roi_feat = fluid.layers.roi_align(
                base_feat,
                rois=rois,
                rois_lod=lod_t,
                pooled_height=cfg.roi_resolution,
                pooled_width=cfg.roi_resolution,
                spatial_scale=cfg.spatial_scale,
                sampling_ratio=cfg.sampling_ratio)

        if cfg.enable_ce:
            print("roi feat: ", np.abs(self.roi_feat.numpy()).mean(),
                  self.roi_feat.shape)
        return roi_feat

    def loss(self):
        losses = []
        # RPN loss
        loss_rpn_cls, loss_rpn_bbox = self.rpn_loss()
        # Fast RCNN loss
        loss_cls, loss_bbox = self.fast_rcnn_loss()
        losses = [loss_cls, loss_bbox, loss_rpn_cls, loss_rpn_bbox]
        if self.cfg.MASK_ON:
            loss_mask_rcnn = self.mask_rcnn_loss()
            losses += [loss_mask_rcnn]

        loss = fluid.layers.sum(losses)
        out = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_bbox': loss_rpn_bbox
        }
        if self.cfg.MASK_ON:
            out['loss_mask_rcnn'] = loss_mask_rcnn
        if self.cfg.enable_ce:
            debug_out = {
                'bbox_pred': self.bbox_pred,
                'cls_score': self.cls_score,
                'roi_feat': self.roi_feat,
                'rpn_cls_score': self.rpn_cls_score,
                'rpn_bbox_pred': self.rpn_bbox_pred,
                'body_conv': self.body_conv,
                'res3': self.res3,
                'res2': self.res2
            }
            out.update(debug_out)
        return out

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
        score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight = rpn_target_assign_pyfn(
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

        # cls loss
        score_tgt = fluid.layers.cast(x=score_tgt, dtype='float32')
        rpn_cls_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=score_pred, label=score_tgt)
        rpn_cls_loss = fluid.layers.reduce_mean(
            rpn_cls_loss, name='loss_rpn_cls')

        # reg loss
        rpn_reg_loss = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=3.0,
            inside_weight=bbox_weight,
            outside_weight=bbox_weight)
        rpn_reg_loss_sum = fluid.layers.reduce_sum(
            rpn_reg_loss, name='loss_rpn_bbox')
        score_shape = score_tgt.shape
        norm = 1
        for i in score_shape:
            norm *= i
        rpn_reg_loss_mean = rpn_reg_loss_sum / norm

        return rpn_cls_loss, rpn_reg_loss_mean

    def fast_rcnn_loss(self):
        labels_int64 = fluid.layers.cast(x=self.labels_int32, dtype='int64')
        labels_int64.stop_gradient = True
        self.cls_score = fluid.layers.reshape(self.cls_score,
                                              (-1, cfg.class_num))
        loss_cls = fluid.layers.softmax_with_cross_entropy(
            logits=self.cls_score,
            label=labels_int64, )
        loss_cls = fluid.layers.reduce_mean(loss_cls)
        loss_bbox = fluid.layers.smooth_l1(
            x=self.bbox_pred,
            y=self.bbox_targets,
            inside_weight=self.bbox_inside_weights,
            outside_weight=self.bbox_outside_weights,
            sigma=1.0)
        loss_bbox = fluid.layers.reduce_mean(loss_bbox)
        return loss_cls, loss_bbox

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

    def eval_model_python(self):
        from .pyops.post_processing import get_nmsed_box
        from functools import reduce
        cls_prob = fluid.layers.softmax(self.cls_score, use_cudnn=False).numpy()
        bbox_pred = self.bbox_pred.numpy()  #.reshape((-1, cfg.class_num, 4))

        img_info = self.im_info.numpy()
        # img_shape = image_info[:2]
        img_scale = img_info[:, 2]
        boxes = self.rpn_rois.numpy()

        # nms box
        new_lod, pred_result = get_nmsed_box(boxes, cls_prob, bbox_pred,
                                             cfg.class_num, img_info)

        self.pred_result = to_variable(pred_result)
        lod_t = to_variable(np.asarray(new_lod))

        if cfg.MASK_ON:
            pred_res_shape = self.pred_result.shape
            shape = reduce((lambda x, y: x * y), pred_res_shape)
            shape = np.asarray(shape).reshape((1, 1))
            ones = np.ones((1, 1), dtype=np.int32)
            cond = (shape == ones).all()
            if cond:
                mask_rcn_logits = pred_result
            else:
                mask_rois = to_variable(pred_result[:, 2:] * img_scale)
                lod_t = to_variable(np.asarray(new_lod))
                mask_roi_feat = self.roi_extractor(self.body_conv, mask_rois,
                                                   lod_t)
                conv5 = self.res5(mask_roi_feat)
                mask_fcn_logits = self.suffixnet(conv5)
            return lod_t, self.pred_result, mask_fcn_logits

        return lod_t, self.pred_result
