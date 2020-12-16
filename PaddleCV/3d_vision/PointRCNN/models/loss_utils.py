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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant

__all__ = ["get_reg_loss"]


def sigmoid_focal_loss(logits, labels, weights, gamma=2.0, alpha=0.25):
    sce_loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, labels)
    prob = fluid.layers.sigmoid(logits)
    p_t = labels * prob + (1.0 - labels) * (1.0 - prob)
    modulating_factor = fluid.layers.pow(1.0 - p_t, gamma)
    alpha_weight_factor = labels * alpha + (1.0 - labels) * (1.0 - alpha)
    return modulating_factor * alpha_weight_factor * sce_loss * weights


def get_reg_loss(pred_reg, reg_label, fg_mask, point_num, loc_scope,
                 loc_bin_size, num_head_bin, anchor_size,
                 get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5,
                 loc_y_bin_size=0.25, get_ry_fine=False):

    """
    Bin-based 3D bounding boxes regression loss. See https://arxiv.org/abs/1812.04244 for more details.

    :param pred_reg: (N, C)
    :param reg_label: (N, 7) [dx, dy, dz, h, w, l, ry]
    :param loc_scope: constant
    :param loc_bin_size: constant
    :param num_head_bin: constant
    :param anchor_size: (N, 3) or (3)
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    fg_num = fluid.layers.cast(fluid.layers.reduce_sum(fg_mask), dtype=pred_reg.dtype)
    fg_num = fluid.layers.clip(fg_num, min=1.0, max=point_num)
    fg_scale = float(point_num) / fg_num

    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    reg_loss_dict = {}

    # xz localization loss
    x_offset_label, y_offset_label, z_offset_label = reg_label[:, 0:1], reg_label[:, 1:2], reg_label[:, 2:3]
    x_shift = fluid.layers.clip(x_offset_label + loc_scope, 0., loc_scope * 2 - 1e-3)
    z_shift = fluid.layers.clip(z_offset_label + loc_scope, 0., loc_scope * 2 - 1e-3)
    x_bin_label = fluid.layers.cast(x_shift / loc_bin_size, dtype='int64')
    z_bin_label = fluid.layers.cast(z_shift / loc_bin_size, dtype='int64')

    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r

    loss_x_bin = fluid.layers.softmax_with_cross_entropy(pred_reg[:, x_bin_l: x_bin_r], x_bin_label)
    loss_x_bin = fluid.layers.reduce_mean(loss_x_bin * fg_mask) * fg_scale
    loss_z_bin = fluid.layers.softmax_with_cross_entropy(pred_reg[:, z_bin_l: z_bin_r], z_bin_label)
    loss_z_bin = fluid.layers.reduce_mean(loss_z_bin * fg_mask) * fg_scale
    reg_loss_dict['loss_x_bin'] = loss_x_bin
    reg_loss_dict['loss_z_bin'] = loss_z_bin
    loc_loss = loss_x_bin + loss_z_bin

    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_label = x_shift - (fluid.layers.cast(x_bin_label, dtype=x_shift.dtype) * loc_bin_size + loc_bin_size / 2.)
        z_res_label = z_shift - (fluid.layers.cast(z_bin_label, dtype=z_shift.dtype) * loc_bin_size + loc_bin_size / 2.)
        x_res_norm_label = x_res_label / loc_bin_size
        z_res_norm_label = z_res_label / loc_bin_size

        x_bin_onehot = fluid.one_hot(x_bin_label[:, 0], depth=per_loc_bin_num)
        z_bin_onehot = fluid.one_hot(z_bin_label[:, 0], depth=per_loc_bin_num)

        loss_x_res = fluid.layers.smooth_l1(fluid.layers.reduce_sum(pred_reg[:, x_res_l: x_res_r] * x_bin_onehot, dim=1, keep_dim=True), x_res_norm_label)
        loss_x_res = fluid.layers.reduce_mean(loss_x_res * fg_mask) * fg_scale
        loss_z_res = fluid.layers.smooth_l1(fluid.layers.reduce_sum(pred_reg[:, z_res_l: z_res_r] * z_bin_onehot, dim=1, keep_dim=True), z_res_norm_label)
        loss_z_res = fluid.layers.reduce_mean(loss_z_res * fg_mask) * fg_scale
        reg_loss_dict['loss_x_res'] = loss_x_res
        reg_loss_dict['loss_z_res'] = loss_z_res
        loc_loss += loss_x_res + loss_z_res

    # y localization loss
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_shift = fluid.layers.clip(y_offset_label + loc_y_scope, 0., loc_y_scope * 2 - 1e-3)
        y_bin_label = fluid.layers.cast(y_shift / loc_y_bin_size, dtype='int64')
        y_res_label = y_shift - (fluid.layers.cast(y_bin_label, dtype=y_shift.dtype) * loc_y_bin_size + loc_y_bin_size / 2.)
        y_res_norm_label = y_res_label / loc_y_bin_size

        y_bin_onehot = fluid.one_hot(y_bin_label[:, 0], depth=per_loc_bin_num)

        loss_y_bin = fluid.layers.cross_entropy(pred_reg[:, y_bin_l: y_bin_r], y_bin_label)
        loss_y_bin = fluid.layers.reduce_mean(loss_y_bin * fg_mask) * fg_scale
        loss_y_res = fluid.layers.smooth_l1(fluid.layers.reduce_sum(pred_reg[:, y_res_l: y_res_r] * y_bin_onehot, dim=1, keep_dim=True), y_res_norm_label)
        loss_y_res = fluid.layers.reduce_mean(loss_y_res * fg_mask) * fg_scale

        reg_loss_dict['loss_y_bin'] = loss_y_bin
        reg_loss_dict['loss_y_res'] = loss_y_res

        loc_loss += loss_y_bin + loss_y_res
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        loss_y_offset = fluid.layers.smooth_l1(fluid.layers.reduce_sum(pred_reg[:, y_offset_l: y_offset_r], dim=1, keep_dim=True), y_offset_label)
        loss_y_offset = fluid.layers.reduce_mean(loss_y_offset * fg_mask) * fg_scale
        reg_loss_dict['loss_y_offset'] = loss_y_offset
        loc_loss += loss_y_offset

    # angle loss
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_label = reg_label[:, 6:7]

    if get_ry_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin

        ry_label = ry_label % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = fluid.layers.logical_and(ry_label > np.pi * 0.5, ry_label < np.pi * 1.5)
        opposite_flag = fluid.layers.cast(opposite_flag, dtype=ry_label.dtype)
        shift_angle = (ry_label + opposite_flag * np.pi + np.pi * 0.5) % (2 * np.pi)  # (0 ~ pi)
        shift_angle.stop_gradient = True

        shift_angle = fluid.layers.clip(shift_angle - np.pi * 0.25, min=1e-3, max=np.pi * 0.5 - 1e-3)  # (0, pi/2)

        # bin center is (5, 10, 15, ..., 85)
        ry_bin_label = fluid.layers.cast(shift_angle / angle_per_class, dtype='int64')
        ry_res_label = shift_angle - (fluid.layers.cast(ry_bin_label, dtype=shift_angle.dtype) * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    else:
        # divide 2pi into several bins
        angle_per_class = (2 * np.pi) / num_head_bin
        heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi

        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        shift_angle.stop_gradient = True
        ry_bin_label = fluid.layers.cast(shift_angle / angle_per_class, dtype='int64')
        ry_res_label = shift_angle - (fluid.layers.cast(ry_bin_label, dtype=shift_angle.dtype) * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    ry_bin_onehot = fluid.one_hot(ry_bin_label[:, 0], depth=num_head_bin)
    loss_ry_bin = fluid.layers.softmax_with_cross_entropy(pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label)
    loss_ry_bin = fluid.layers.reduce_mean(loss_ry_bin * fg_mask) * fg_scale
    loss_ry_res = fluid.layers.smooth_l1(fluid.layers.reduce_sum(pred_reg[:, ry_res_l: ry_res_r] * ry_bin_onehot, dim=1, keep_dim=True), ry_res_norm_label)
    loss_ry_res = fluid.layers.reduce_mean(loss_ry_res * fg_mask) * fg_scale

    reg_loss_dict['loss_ry_bin'] = loss_ry_bin
    reg_loss_dict['loss_ry_res'] = loss_ry_res
    angle_loss = loss_ry_bin + loss_ry_res

    # size loss
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert pred_reg.shape[1] == size_res_r, '%d vs %d' % (pred_reg.shape[1], size_res_r)

    anchor_size_var = fluid.layers.zeros(shape=[3], dtype=reg_label.dtype)
    fluid.layers.assign(np.array(anchor_size).astype('float32'), anchor_size_var)
    size_res_norm_label = (reg_label[:, 3:6] - anchor_size_var) / anchor_size_var
    size_res_norm_label = fluid.layers.reshape(size_res_norm_label, shape=[-1, 1], inplace=True)
    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    size_res_norm = fluid.layers.reshape(size_res_norm, shape=[-1, 1], inplace=True)
    size_loss = fluid.layers.smooth_l1(size_res_norm, size_res_norm_label)
    size_loss = fluid.layers.reshape(size_loss, shape=[-1, 3])
    size_loss = fluid.layers.reduce_mean(size_loss * fg_mask) * fg_scale

    # Total regression loss
    reg_loss_dict['loss_loc'] = loc_loss
    reg_loss_dict['loss_angle'] = angle_loss
    reg_loss_dict['loss_size'] = size_loss

    return loc_loss, angle_loss, size_loss, reg_loss_dict

