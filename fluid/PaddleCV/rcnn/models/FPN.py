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
import collections
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import Xavier
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Normal
from config import cfg

#LOWEST_BACKBONE_LVL = 2   
#HIGHEST_BACKBONE_LVL = 5 
#res_name = ['res5c_sum', 'res4f_sum', 'res3d_sum', 'res2c_sum']


def add_fpn_onto_conv_body(res_dict, res_name_list):
    spatial_scales = [1. / 32., 1. / 16., 1. / 8., 1. / 4.]
    fpn_dim = cfg.FPN_dim
    max_level = max(cfg.FPN_rpn_max_level, cfg.FPN_roi_max_level)
    min_level = min(cfg.FPN_rpn_min_level, cfg.FPN_roi_min_level)
    num_backbone_stages = len(res_name_list) - (min_level -
                                                cfg.LOWEST_BACKBONE_LVL)
    res_name_list = res_name_list[:num_backbone_stages]
    spatial_scales = spatial_scales[:num_backbone_stages]
    fpn_inner_output = [[] for _ in range(num_backbone_stages)]
    fpn_inner_name = 'fpn_inner_' + res_name_list[0]
    fpn_inner_output[0] = fluid.layers.conv2d(
        res_dict[res_name_list[0]],
        fpn_dim,
        1,
        param_attr=ParamAttr(
            name=fpn_inner_name + "_w", initializer=Xavier()),
        bias_attr=ParamAttr(
            name=fpn_inner_name + "_b",
            learning_rate=2.,
            regularizer=L2Decay(0.)),
        name=fpn_inner_name)
    for i in range(1, num_backbone_stages):
        add_topdown_lateral_module(
            i,
            res_dict,
            res_name_list,
            fpn_inner_output,
            fpn_dim, )
    fpn_dict = {}
    fpn_name_list = []

    for i in range(num_backbone_stages):
        fpn_name = 'fpn_' + res_name_list[i]
        fpn_output = fluid.layers.conv2d(
            fpn_inner_output[i],
            fpn_dim,
            filter_size=3,
            padding=1,
            param_attr=ParamAttr(
                name=fpn_name + "_w", initializer=Xavier()),
            bias_attr=ParamAttr(
                name=fpn_name + "_b", learning_rate=2.,
                regularizer=L2Decay(0.)),
            name=fpn_name)
        fpn_dict[fpn_name] = fpn_output
        fpn_name_list.append(fpn_name)
    if max_level == cfg.HIGHEST_BACKBONE_LVL + 1:
        P6_input_name = fpn_name_list[0]
        P6_output = fluid.layers.pool2d(
            fpn_dict[P6_input_name],
            1,
            'max',
            pool_stride=2,
            name=P6_input_name + '_subsampled_2x')
        fpn_dict[P6_output.name] = P6_output
        fpn_name_list.insert(0, P6_output.name)
        spatial_scales.insert(0, spatial_scales[0] * 0.5)
    return fpn_dict, spatial_scales, fpn_name_list


def add_topdown_lateral_module(index, res_dict, res_name_list, fpn_inner_output,
                               fpn_dim):
    lateral_name = 'fpn_inner_' + res_name_list[index] + '_lateral'
    topdown_name = 'fpn_topdown_' + res_name_list[index]
    fpn_inner_name = 'fpn_inner_' + res_name_list[index]
    lateral = fluid.layers.conv2d(
        res_dict[res_name_list[index]],
        fpn_dim,
        1,
        param_attr=ParamAttr(
            name=lateral_name + "_w", initializer=Xavier()),
        bias_attr=ParamAttr(
            name=lateral_name + "_b", learning_rate=2.,
            regularizer=L2Decay(0.)),
        name=lateral_name)
    shape = fluid.layers.shape(fpn_inner_output[index - 1])
    shape_hw = fluid.layers.slice(shape, axes=[0], starts=[2], ends=[4])
    shape_hw.stop_gradient = True
    in_shape = fluid.layers.cast(shape_hw, dtype='int32')
    out_shape = in_shape * 2
    out_shape.stop_gradient = True
    td = fluid.layers.resize_nearest(
        fpn_inner_output[index - 1],
        scale=2.,
        actual_shape=out_shape,
        name=topdown_name)

    fpn_inner_output[index] = fluid.layers.elementwise_add(
        lateral, td, name=fpn_inner_name)


def add_fpn_rpn_outputs(fpn_dict, im_info, fpn_name_list, mode):
    num_anchors = len(cfg.FPN_rpn_aspect_ratios)
    k_max = cfg.FPN_rpn_max_level
    k_min = cfg.FPN_rpn_min_level
    fpn_dim = cfg.FPN_dim
    sk_min = str(k_min)
    conv_share_name = 'conv_rpn_fpn' + sk_min
    cls_share_name = 'rpn_cls_logits_fpn' + sk_min
    bbox_share_name = 'rpn_bbox_pred_fpn' + sk_min
    rpn_fpn_list = []
    rpn_rois_list = []
    rpn_roi_probs_list = []
    anchors_list = []
    anchor_num_list = []
    var_list = []
    for lvl in range(k_min, k_max + 1):
        input_name = fpn_name_list[k_max - lvl]
        slvl = str(lvl)
        conv_name = 'conv_rpn_fpn' + slvl
        cls_name = 'rpn_cls_logits_fpn' + slvl
        bbox_name = 'rpn_bbox_pred_fpn' + slvl
        conv_rpn_fpn = fluid.layers.conv2d(
            input=fpn_dict[input_name],
            num_filters=fpn_dim,
            filter_size=3,
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
        conv_rpn_fpn.persistable = True
        rpn_cls_logits_fpn = fluid.layers.conv2d(
            input=conv_rpn_fpn,
            num_filters=num_anchors,
            filter_size=1,
            act=None,
            name=cls_name,
            param_attr=ParamAttr(
                name=cls_share_name + '_w',
                initializer=Normal(
                    loc=0., scale=0.01)),
            bias_attr=ParamAttr(
                name=cls_share_name + '_b',
                learning_rate=2.,
                regularizer=L2Decay(0.)))
        shape = fluid.layers.shape(rpn_cls_logits_fpn)
        shape_chw = fluid.layers.slice(shape, axes=[0], starts=[1], ends=[4])
        anchors_num = fluid.layers.reduce_prod(shape_chw)
        if lvl == k_min:
            anchor_num_list.append(anchors_num)
        else:
            anchor_num_list.append(anchor_num_list[-1] + anchors_num)
        anchor_num_list[-1].stop_gradient = True
        rpn_cls_logits_fpn.persistable = True
        rpn_bbox_pred_fpn = fluid.layers.conv2d(
            input=conv_rpn_fpn,
            num_filters=num_anchors * 4,
            filter_size=1,
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
        rpn_bbox_pred_fpn.persistable = True
        rpn_fpn_list.append((rpn_cls_logits_fpn, rpn_bbox_pred_fpn))

        anchors, var = fluid.layers.anchor_generator(
            input=conv_rpn_fpn,
            anchor_sizes=(cfg.FPN_rpn_anchor_start_size * 2.**(lvl - k_min), ),
            aspect_ratios=cfg.FPN_rpn_aspect_ratios,
            variance=cfg.variances,
            stride=(2.**lvl, 2.**lvl))

        rpn_cls_probs_fpn = fluid.layers.sigmoid(
            rpn_cls_logits_fpn, name='rpn_cls_probs_fpn' + slvl)
        rpn_cls_probs_fpn.persistable = True

        param_obj = cfg.TRAIN if mode == 'train' else cfg.TEST
        pre_nms_top_n = param_obj.rpn_pre_nms_top_n
        post_nms_top_n = param_obj.rpn_post_nms_top_n
        nms_thresh = param_obj.rpn_nms_thresh
        min_size = param_obj.rpn_min_size
        eta = param_obj.rpn_eta

        rpn_rois_fpn, rpn_roi_probs_fpn = fluid.layers.generate_proposals(
            scores=rpn_cls_probs_fpn,
            bbox_deltas=rpn_bbox_pred_fpn,
            im_info=im_info,
            anchors=anchors,
            variances=var,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=nms_thresh,
            min_size=min_size,
            eta=eta)
        rpn_rois_fpn.persistable = True
        rpn_roi_probs_fpn.persistable = True
        rpn_rois_list.append(rpn_rois_fpn)
        rpn_roi_probs_list.append(rpn_roi_probs_fpn)
        anchors_list.append(anchors)
        var_list.append(var)
    return rpn_fpn_list, rpn_rois_list, rpn_roi_probs_list, anchors_list, var_list, anchor_num_list


def add_FPN_roi_head(head_inputs, rois_list, fpn_name_list, spatial_scale):
    k_max = cfg.FPN_roi_max_level
    k_min = cfg.FPN_roi_min_level
    num_roi_lvls = k_max - k_min + 1
    input_name_list = fpn_name_list[-num_roi_lvls:]
    spatial_scale = spatial_scale[-num_roi_lvls:]
    roi_out_list = []
    for lvl in range(k_min, k_max + 1):
        rois = rois_list[lvl - k_min]
        input_name = input_name_list[k_max - lvl]
        head_input = head_inputs[input_name]
        sc = spatial_scale[k_max - lvl]
        if cfg.roi_func == 'RoIPool':
            roi_out = fluid.layers.roi_pool(
                input=head_input,
                rois=rois,
                pooled_height=cfg.roi_resolution,
                pooled_width=cfg.roi_resolution,
                spatial_scale=sc,
                name='roi_pool_lvl_' + str(lvl))
        elif cfg.roi_func == 'RoIAlign':
            roi_out = fluid.layers.roi_align(
                input=head_input,
                rois=rois,
                pooled_height=cfg.roi_resolution,
                pooled_width=cfg.roi_resolution,
                spatial_scale=sc,
                sampling_ratio=cfg.sampling_ratio,
                name='roi_align_lvl_' + str(lvl))
        roi_out_list.append(roi_out)
        roi_out.persistable = True

    return roi_out_list


def add_FPN_roi_head_output(body_dict, pool_rois, body_name_list,
                            spatial_scale):
    rois_collect = pool_rois[0]
    rois = pool_rois[1]
    restore_index = pool_rois[2]
    roi_out_list = add_FPN_roi_head(body_dict, rois, body_name_list,
                                    spatial_scale)
    roi_feat_shuffle = fluid.layers.concat(roi_out_list)
    roi_feat = fluid.layers.gather(roi_feat_shuffle, restore_index)
    roi_feat = fluid.layers.lod_reset(roi_feat, rois_collect)
    roi_feat.persistable = True
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
    fc7.persistable = True
    return roi_feat, fc7
