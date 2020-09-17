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

import paddle
import paddle.nn.functional as F
from paddle import ParamAttr
import numpy as np
import math

from bmn_utils import get_interp1d_mask

DATATYPE = 'float32'


# Net
class Conv1D(paddle.nn.Layer):
    def __init__(self,
                 prefix,
                 num_channels=256,
                 num_filters=256,
                 size_k=3,
                 padding=1,
                 groups=1,
                 act="relu"):
        super(Conv1D, self).__init__()
        fan_in = num_channels * size_k * 1
        k = 1. / math.sqrt(fan_in)
        param_attr = ParamAttr(
            name=prefix + "_w",
            initializer=paddle.nn.initializer.Uniform(
                low=-k, high=k))
        bias_attr = ParamAttr(
            name=prefix + "_b",
            initializer=paddle.nn.initializer.Uniform(
                low=-k, high=k))

        self._conv2d = paddle.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=(1, size_k),
            stride=1,
            padding=(0, padding),
            groups=groups,
            weight_attr=param_attr,
            bias_attr=bias_attr)
        if act == "relu":
            self._act = paddle.nn.ReLU()
        elif act == "sigmoid":
            self._act = paddle.nn.Sigmoid()

    def forward(self, x):
        x = paddle.unsqueeze(x, axis=[2])
        x = self._conv2d(x)
        x = self._act(x)
        x = paddle.squeeze(x, axis=[2])
        return x


class BMN(paddle.nn.Layer):
    def __init__(self, cfg):
        super(BMN, self).__init__()

        #init config
        self.tscale = cfg.MODEL.tscale
        self.dscale = cfg.MODEL.dscale
        self.prop_boundary_ratio = cfg.MODEL.prop_boundary_ratio
        self.num_sample = cfg.MODEL.num_sample
        self.num_sample_perbin = cfg.MODEL.num_sample_perbin

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        # Base Module
        self.b_conv1 = Conv1D(
            prefix="Base_1",
            num_channels=400,
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")
        self.b_conv2 = Conv1D(
            prefix="Base_2",
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")

        # Temporal Evaluation Module
        self.ts_conv1 = Conv1D(
            prefix="TEM_s1",
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")
        self.ts_conv2 = Conv1D(
            prefix="TEM_s2", num_filters=1, size_k=1, padding=0, act="sigmoid")
        self.te_conv1 = Conv1D(
            prefix="TEM_e1",
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")
        self.te_conv2 = Conv1D(
            prefix="TEM_e2", num_filters=1, size_k=1, padding=0, act="sigmoid")

        #Proposal Evaluation Module
        self.p_conv1 = Conv1D(
            prefix="PEM_1d",
            num_filters=self.hidden_dim_2d,
            size_k=3,
            padding=1,
            act="relu")

        # init to speed up
        sample_mask = get_interp1d_mask(self.tscale, self.dscale,
                                        self.prop_boundary_ratio,
                                        self.num_sample, self.num_sample_perbin)
        self.sample_mask = paddle.to_tensor(sample_mask)
        self.sample_mask.stop_gradient = True

        self.p_conv3d1 = paddle.nn.Conv3d(
            in_channels=128,
            out_channels=self.hidden_dim_3d,
            kernel_size=(self.num_sample, 1, 1),
            stride=(self.num_sample, 1, 1),
            padding=0,
            weight_attr=ParamAttr(name="PEM_3d1_w"),
            bias_attr=ParamAttr(name="PEM_3d1_b"))
        self.p_conv3d1_act = paddle.nn.ReLU()

        self.p_conv2d1 = paddle.nn.Conv2d(
            in_channels=512,
            out_channels=self.hidden_dim_2d,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name="PEM_2d1_w"),
            bias_attr=ParamAttr(name="PEM_2d1_b"))
        self.p_conv2d1_act = paddle.nn.ReLU()

        self.p_conv2d2 = paddle.nn.Conv2d(
            in_channels=128,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(name="PEM_2d2_w"),
            bias_attr=ParamAttr(name="PEM_2d2_b"))
        self.p_conv2d2_act = paddle.nn.ReLU()

        self.p_conv2d3 = paddle.nn.Conv2d(
            in_channels=128,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(name="PEM_2d3_w"),
            bias_attr=ParamAttr(name="PEM_2d3_b"))
        self.p_conv2d3_act = paddle.nn.ReLU()

        self.p_conv2d4 = paddle.nn.Conv2d(
            in_channels=128,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name="PEM_2d4_w"),
            bias_attr=ParamAttr(name="PEM_2d4_b"))
        self.p_conv2d4_act = paddle.nn.Sigmoid()

    def forward(self, x):
        #Base Module
        x = self.b_conv1(x)
        x = self.b_conv2(x)

        #TEM
        xs = self.ts_conv1(x)
        xs = self.ts_conv2(xs)
        xs = paddle.squeeze(xs, axis=[1])
        xe = self.te_conv1(x)
        xe = self.te_conv2(xe)
        xe = paddle.squeeze(xe, axis=[1])

        #PEM
        xp = self.p_conv1(x)
        #BM layer
        xp = paddle.matmul(xp, self.sample_mask)
        xp = paddle.reshape(xp, shape=[0, 0, -1, self.dscale, self.tscale])

        xp = self.p_conv3d1(xp)
        xp = self.p_conv3d1_act(xp)
        xp = paddle.squeeze(xp, axis=[2])
        xp = self.p_conv2d1(xp)
        xp = self.p_conv2d1_act(xp)
        xp = self.p_conv2d2(xp)
        xp = self.p_conv2d2_act(xp)
        xp = self.p_conv2d3(xp)
        xp = self.p_conv2d3_act(xp)
        xp = self.p_conv2d4(xp)
        xp = self.p_conv2d4_act(xp)
        return xp, xs, xe


def bmn_loss_func(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end,
                  cfg):
    def _get_mask(cfg):
        dscale = cfg.MODEL.dscale
        tscale = cfg.MODEL.tscale
        bm_mask = []
        for idx in range(dscale):
            mask_vector = [1 for i in range(tscale - idx)
                           ] + [0 for i in range(idx)]
            bm_mask.append(mask_vector)
        bm_mask = np.array(bm_mask, dtype=np.float32)
        bm_mask = paddle.to_tensor(bm_mask)
        bm_mask.stop_gradient = True
        return bm_mask

    def tem_loss_func(pred_start, pred_end, gt_start, gt_end):
        def bi_loss(pred_score, gt_label):
            pred_score = paddle.reshape(x=pred_score, shape=[-1])
            gt_label = paddle.reshape(x=gt_label, shape=[-1])
            gt_label.stop_gradient = True
            pmask = paddle.cast(x=(gt_label > 0.5), dtype=DATATYPE)
            num_entries = paddle.cast(paddle.shape(pmask), dtype=DATATYPE)
            num_positive = paddle.cast(paddle.reduce_sum(pmask), dtype=DATATYPE)
            ratio = num_entries / num_positive
            coef_0 = 0.5 * ratio / (ratio - 1)
            coef_1 = 0.5 * ratio
            epsilon = 0.000001
            temp = paddle.log(pred_score + epsilon)
            loss_pos = paddle.multiply(paddle.log(pred_score + epsilon), pmask)
            loss_pos = coef_1 * paddle.reduce_mean(loss_pos)
            loss_neg = paddle.multiply(
                paddle.log(1.0 - pred_score + epsilon), (1.0 - pmask))
            loss_neg = coef_0 * paddle.reduce_mean(loss_neg)
            loss = -1 * (loss_pos + loss_neg)
            return loss

        loss_start = bi_loss(pred_start, gt_start)
        loss_end = bi_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss

    def pem_reg_loss_func(pred_score, gt_iou_map, mask):

        gt_iou_map = paddle.multiply(gt_iou_map, mask)

        u_hmask = paddle.cast(x=gt_iou_map > 0.7, dtype=DATATYPE)
        u_mmask = paddle.logical_and(gt_iou_map <= 0.7, gt_iou_map > 0.3)
        u_mmask = paddle.cast(x=u_mmask, dtype=DATATYPE)
        u_lmask = paddle.logical_and(gt_iou_map <= 0.3, gt_iou_map >= 0.)
        u_lmask = paddle.cast(x=u_lmask, dtype=DATATYPE)
        u_lmask = paddle.multiply(u_lmask, mask)

        num_h = paddle.cast(paddle.reduce_sum(u_hmask), dtype=DATATYPE)
        num_m = paddle.cast(paddle.reduce_sum(u_mmask), dtype=DATATYPE)
        num_l = paddle.cast(paddle.reduce_sum(u_lmask), dtype=DATATYPE)

        r_m = num_h / num_m
        u_smmask = paddle.uniform(
            shape=[gt_iou_map.shape[1], gt_iou_map.shape[2]],
            dtype=DATATYPE,
            min=0.0,
            max=1.0)
        u_smmask = paddle.multiply(u_mmask, u_smmask)
        u_smmask = paddle.cast(x=(u_smmask > (1. - r_m)), dtype=DATATYPE)

        r_l = num_h / num_l
        u_slmask = paddle.uniform(
            shape=[gt_iou_map.shape[1], gt_iou_map.shape[2]],
            dtype=DATATYPE,
            min=0.0,
            max=1.0)
        u_slmask = paddle.multiply(u_lmask, u_slmask)
        u_slmask = paddle.cast(x=(u_slmask > (1. - r_l)), dtype=DATATYPE)

        weights = u_hmask + u_smmask + u_slmask
        weights.stop_gradient = True
        loss = F.square_error_cost(pred_score, gt_iou_map)
        loss = paddle.multiply(loss, weights)
        loss = 0.5 * paddle.reduce_sum(loss) / paddle.reduce_sum(weights)

        return loss

    def pem_cls_loss_func(pred_score, gt_iou_map, mask):
        gt_iou_map = paddle.multiply(gt_iou_map, mask)
        gt_iou_map.stop_gradient = True
        pmask = paddle.cast(x=(gt_iou_map > 0.9), dtype=DATATYPE)
        nmask = paddle.cast(x=(gt_iou_map <= 0.9), dtype=DATATYPE)
        nmask = paddle.multiply(nmask, mask)

        num_positive = paddle.reduce_sum(pmask)
        num_entries = num_positive + paddle.reduce_sum(nmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = paddle.multiply(paddle.log(pred_score + epsilon), pmask)
        loss_pos = coef_1 * paddle.reduce_sum(loss_pos)
        loss_neg = paddle.multiply(
            paddle.log(1.0 - pred_score + epsilon), nmask)
        loss_neg = coef_0 * paddle.reduce_sum(loss_neg)
        loss = -1 * (loss_pos + loss_neg) / num_entries
        return loss

    pred_bm_reg = paddle.squeeze(
        paddle.slice(
            pred_bm, axes=[1], starts=[0], ends=[1]), axis=[1])
    pred_bm_cls = paddle.squeeze(
        paddle.slice(
            pred_bm, axes=[1], starts=[1], ends=[2]), axis=[1])

    bm_mask = _get_mask(cfg)

    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)

    tem_loss = tem_loss_func(pred_start, pred_end, gt_start, gt_end)

    loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
    return loss, tem_loss, pem_reg_loss, pem_cls_loss
