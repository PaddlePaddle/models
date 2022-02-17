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
from paddle.fluid import ParamAttr
import numpy as np
import math

DATATYPE = 'float32'


class BMN_NET(object):
    def __init__(self, mode, cfg):
        self.tscale = cfg["tscale"]
        self.dscale = cfg["dscale"]
        self.prop_boundary_ratio = cfg["prop_boundary_ratio"]
        self.num_sample = cfg["num_sample"]
        self.num_sample_perbin = cfg["num_sample_perbin"]
        self.is_training = (mode == 'train')

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()
        self._get_mask()

    def conv1d(self,
               input,
               num_k=256,
               input_size=256,
               size_k=3,
               padding=1,
               groups=1,
               act='relu',
               name="conv1d"):
        fan_in = input_size * size_k * 1
        k = 1. / math.sqrt(fan_in)
        param_attr = fluid.initializer.Uniform(low=-k, high=k)
        bias_attr = fluid.initializer.Uniform(low=-k, high=k)

        input = fluid.layers.unsqueeze(input=input, axes=[2])
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_k,
            filter_size=(1, size_k),
            stride=1,
            padding=(0, padding),
            groups=groups,
            act=act,
            name=name,
            param_attr=param_attr,
            bias_attr=bias_attr)
        conv = fluid.layers.squeeze(input=conv, axes=[2])
        return conv

    def conv2d(self,
               input,
               num_k=256,
               size_k=3,
               padding=1,
               act='relu',
               name='conv2d'):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_k,
            filter_size=size_k,
            stride=1,
            padding=padding,
            act=act,
            name=name)
        return conv

    def conv3d(self, input, num_k=512, name="PEM_3d"):
        conv = fluid.layers.conv3d(
            input=input,
            num_filters=num_k,
            filter_size=(self.num_sample, 1, 1),
            stride=(self.num_sample, 1, 1),
            padding=0,
            act='relu',
            name=name)
        return conv

    def net(self, input):
        # Base Module of BMN
        x_1d = self.conv1d(
            input,
            input_size=400,
            num_k=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu",
            name="Base_1")
        x_1d = self.conv1d(
            x_1d,
            num_k=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu",
            name="Base_2")

        # Temporal Evaluation Module of BMN
        x_1d_s = self.conv1d(
            x_1d,
            num_k=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu",
            name="TEM_s1")
        x_1d_s = self.conv1d(
            x_1d_s, num_k=1, size_k=1, padding=0, act="sigmoid", name="TEM_s2")
        x_1d_e = self.conv1d(
            x_1d,
            num_k=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu",
            name="TEM_e1")
        x_1d_e = self.conv1d(
            x_1d_e, num_k=1, size_k=1, padding=0, act="sigmoid", name="TEM_e2")
        x_1d_s = fluid.layers.squeeze(input=x_1d_s, axes=[1])
        x_1d_e = fluid.layers.squeeze(input=x_1d_e, axes=[1])

        # Proposal Evaluation Module of BMN
        x_1d = self.conv1d(
            x_1d,
            num_k=self.hidden_dim_2d,
            size_k=3,
            padding=1,
            act="relu",
            name="PEM_1d")
        x_3d = self._boundary_matching_layer(x_1d)
        x_3d = self.conv3d(x_3d, self.hidden_dim_3d, name="PEM_3d1")

        x_2d = fluid.layers.squeeze(input=x_3d, axes=[2])
        x_2d = self.conv2d(
            x_2d,
            self.hidden_dim_2d,
            size_k=1,
            padding=0,
            act='relu',
            name="PEM_2d1")
        x_2d = self.conv2d(
            x_2d,
            self.hidden_dim_2d,
            size_k=3,
            padding=1,
            act='relu',
            name="PEM_2d2")
        x_2d = self.conv2d(
            x_2d,
            self.hidden_dim_2d,
            size_k=3,
            padding=1,
            act='relu',
            name="PEM_2d3")
        x_2d = self.conv2d(
            x_2d, 2, size_k=1, padding=0, act='sigmoid', name="PEM_2d4")
        return x_2d, x_1d_s, x_1d_e

    def _get_mask(self):
        bm_mask = []
        for idx in range(self.dscale):
            mask_vector = [1 for i in range(self.tscale - idx)
                           ] + [0 for i in range(idx)]
            bm_mask.append(mask_vector)
        bm_mask = np.array(bm_mask, dtype=np.float32)
        self.bm_mask = fluid.layers.create_global_var(
            shape=[self.dscale, self.tscale],
            value=0,
            dtype=DATATYPE,
            persistable=True)
        fluid.layers.assign(bm_mask, self.bm_mask)
        self.bm_mask.stop_gradient = True

    def _boundary_matching_layer(self, x):
        out = fluid.layers.matmul(x, self.sample_mask)
        out = fluid.layers.reshape(
            x=out, shape=[0, 0, -1, self.dscale, self.tscale])
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample,
                               num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) *
                                        num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for start_index in range(self.tscale):
            mask_mat_vector = []
            for duration_index in range(self.dscale):
                if start_index + duration_index < self.tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)

        self.sample_mask = fluid.layers.create_parameter(
            shape=[self.tscale, self.num_sample, self.dscale, self.tscale],
            dtype=DATATYPE,
            attr=fluid.ParamAttr(
                name="sample_mask", trainable=False),
            default_initializer=fluid.initializer.NumpyArrayInitializer(
                mask_mat))
        self.sample_mask = fluid.layers.reshape(
            x=self.sample_mask, shape=[self.tscale, -1], inplace=True)
        self.sample_mask.stop_gradient = True

    def tem_loss_func(self, pred_start, pred_end, gt_start, gt_end):
        def bi_loss(pred_score, gt_label):
            pred_score = fluid.layers.reshape(
                x=pred_score, shape=[-1], inplace=True)
            gt_label = fluid.layers.reshape(
                x=gt_label, shape=[-1], inplace=False)
            gt_label.stop_gradient = True
            pmask = fluid.layers.cast(x=(gt_label > 0.5), dtype=DATATYPE)
            num_entries = fluid.layers.cast(
                fluid.layers.shape(pmask), dtype=DATATYPE)
            num_positive = fluid.layers.cast(
                fluid.layers.reduce_sum(pmask), dtype=DATATYPE)
            ratio = num_entries / num_positive
            coef_0 = 0.5 * ratio / (ratio - 1)
            coef_1 = 0.5 * ratio
            epsilon = 0.000001
            loss_pos = fluid.layers.elementwise_mul(
                fluid.layers.log(pred_score + epsilon), pmask)
            loss_pos = coef_1 * fluid.layers.reduce_mean(loss_pos)
            loss_neg = fluid.layers.elementwise_mul(
                fluid.layers.log(1.0 - pred_score + epsilon), (1.0 - pmask))
            loss_neg = coef_0 * fluid.layers.reduce_mean(loss_neg)
            loss = -1 * (loss_pos + loss_neg)
            return loss

        loss_start = bi_loss(pred_start, gt_start)
        loss_end = bi_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss

    def pem_reg_loss_func(self, pred_score, gt_iou_map, mask):

        gt_iou_map = fluid.layers.elementwise_mul(gt_iou_map, mask)

        u_hmask = fluid.layers.cast(x=gt_iou_map > 0.7, dtype=DATATYPE)
        u_mmask = fluid.layers.logical_and(gt_iou_map <= 0.7, gt_iou_map > 0.3)
        u_mmask = fluid.layers.cast(x=u_mmask, dtype=DATATYPE)
        u_lmask = fluid.layers.logical_and(gt_iou_map <= 0.3, gt_iou_map >= 0.)
        u_lmask = fluid.layers.cast(x=u_lmask, dtype=DATATYPE)
        u_lmask = fluid.layers.elementwise_mul(u_lmask, mask)

        num_h = fluid.layers.cast(
            fluid.layers.reduce_sum(u_hmask), dtype=DATATYPE)
        num_m = fluid.layers.cast(
            fluid.layers.reduce_sum(u_mmask), dtype=DATATYPE)
        num_l = fluid.layers.cast(
            fluid.layers.reduce_sum(u_lmask), dtype=DATATYPE)

        r_m = num_h / num_m
        u_smmask = fluid.layers.uniform_random(
            shape=[gt_iou_map.shape[1], gt_iou_map.shape[2]],
            dtype=DATATYPE,
            min=0.0,
            max=1.0)
        u_smmask = fluid.layers.elementwise_mul(u_mmask, u_smmask)
        u_smmask = fluid.layers.cast(x=(u_smmask > (1. - r_m)), dtype=DATATYPE)

        r_l = num_h / num_l
        u_slmask = fluid.layers.uniform_random(
            shape=[gt_iou_map.shape[1], gt_iou_map.shape[2]],
            dtype=DATATYPE,
            min=0.0,
            max=1.0)
        u_slmask = fluid.layers.elementwise_mul(u_lmask, u_slmask)
        u_slmask = fluid.layers.cast(x=(u_slmask > (1. - r_l)), dtype=DATATYPE)

        weights = u_hmask + u_smmask + u_slmask
        weights.stop_gradient = True
        loss = fluid.layers.square_error_cost(pred_score, gt_iou_map)
        loss = fluid.layers.elementwise_mul(loss, weights)
        loss = 0.5 * fluid.layers.reduce_sum(loss) / fluid.layers.reduce_sum(
            weights)
        return loss

    def pem_cls_loss_func(self, pred_score, gt_iou_map, mask):
        gt_iou_map = fluid.layers.elementwise_mul(gt_iou_map, mask)
        gt_iou_map.stop_gradient = True
        pmask = fluid.layers.cast(x=(gt_iou_map > 0.9), dtype=DATATYPE)
        nmask = fluid.layers.cast(x=(gt_iou_map <= 0.9), dtype=DATATYPE)
        nmask = fluid.layers.elementwise_mul(nmask, mask)

        num_positive = fluid.layers.reduce_sum(pmask)
        num_entries = num_positive + fluid.layers.reduce_sum(nmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = fluid.layers.elementwise_mul(
            fluid.layers.log(pred_score + epsilon), pmask)
        loss_pos = coef_1 * fluid.layers.reduce_sum(loss_pos)
        loss_neg = fluid.layers.elementwise_mul(
            fluid.layers.log(1.0 - pred_score + epsilon), nmask)
        loss_neg = coef_0 * fluid.layers.reduce_sum(loss_neg)
        loss = -1 * (loss_pos + loss_neg) / num_entries
        return loss

    def bmn_loss_func(self, pred_bm, pred_start, pred_end, gt_iou_map, gt_start,
                      gt_end, bm_mask):
        pred_bm_reg = fluid.layers.squeeze(
            fluid.layers.slice(
                pred_bm, axes=[1], starts=[0], ends=[1]),
            axes=[1])
        pred_bm_cls = fluid.layers.squeeze(
            fluid.layers.slice(
                pred_bm, axes=[1], starts=[1], ends=[2]),
            axes=[1])

        pem_reg_loss = self.pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
        pem_cls_loss = self.pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)
        tem_loss = self.tem_loss_func(pred_start, pred_end, gt_start, gt_end)

        loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
        return loss, tem_loss, pem_reg_loss, pem_cls_loss
