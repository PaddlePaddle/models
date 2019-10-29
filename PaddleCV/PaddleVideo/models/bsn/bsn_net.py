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
#from .ctcn_utils import get_ctcn_conv_initializer as get_init
import math

DATATYPE = 'float32'


class BsnTemNet(object):
    def __init__(self, cfg):
        self.tscale = cfg["tscale"]
        self.feat_dim = cfg["feat_dim"]
        self.hidden_dim = cfg["hidden_dim"]

    def conv1d(self,
               input,
               num_k=256,
               input_size=256,
               size_k=3,
               padding=1,
               act='relu',
               name="conv1d"):
        fan_in = input_size * size_k * 1
        k = 1. / math.sqrt(fan_in)
        param_attr = fluid.ParamAttr(initializer=fluid.initializer.Uniform(
            low=-k, high=k))
        bias_attr = fluid.ParamAttr(initializer=fluid.initializer.Uniform(
            low=-k, high=k))
        input = fluid.layers.unsqueeze(input=input, axes=[2])
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_k,
            filter_size=(1, size_k),
            stride=1,
            padding=(0, padding),
            act=act,
            name=name,
            param_attr=param_attr,
            bias_attr=bias_attr)
        conv = fluid.layers.squeeze(input=conv, axes=[2])
        return conv

    def net(self, input):
        x_1d = self.conv1d(
            input,
            input_size=self.feat_dim,
            num_k=self.hidden_dim,
            size_k=3,
            padding=1,
            act="relu",
            name="Base_1")
        x_1d = self.conv1d(
            x_1d,
            input_size=self.hidden_dim,
            num_k=self.hidden_dim,
            size_k=3,
            padding=1,
            act="relu",
            name="Base_2")
        x_1d = self.conv1d(
            x_1d,
            input_size=self.hidden_dim,
            num_k=3,
            size_k=1,
            padding=0,
            act="sigmoid",
            name="Pred")
        return x_1d

    def loss_func(self, preds, gt_start, gt_end, gt_action):
        pred_start = fluid.layers.squeeze(
            fluid.layers.slice(
                preds, axes=[1], starts=[0], ends=[1]), axes=[1])
        pred_end = fluid.layers.squeeze(
            fluid.layers.slice(
                preds, axes=[1], starts=[1], ends=[2]), axes=[1])
        pred_action = fluid.layers.squeeze(
            fluid.layers.slice(
                preds, axes=[1], starts=[2], ends=[3]), axes=[1])

        def bi_loss(pred_score, gt_label):
            pred_score = fluid.layers.reshape(
                x=pred_score, shape=[-1], inplace=True)
            gt_label = fluid.layers.reshape(
                x=gt_label, shape=[-1], inplace=False)
            gt_label.stop_gradient = True
            pmask = fluid.layers.cast(x=(gt_label > 0.5), dtype='float32')
            num_entries = fluid.layers.cast(
                fluid.layers.shape(pmask), dtype='float32')
            num_positive = fluid.layers.cast(
                fluid.layers.reduce_sum(pmask), dtype='float32')
            ratio = num_entries / num_positive
            coef_0 = 0.5 * num_entries / (num_entries - num_positive + 1)
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
        loss_action = bi_loss(pred_action, gt_action)
        loss = loss_start + loss_end + loss_action
        return loss, loss_start, loss_end, loss_action


class BsnPemNet(object):
    def __init__(self, cfg):
        self.feat_dim = cfg["feat_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.batch_size = cfg["batch_size"]
        self.top_K = cfg["top_K"]
        self.num_gpus = cfg["num_gpus"]
        self.mini_batch = self.batch_size // self.num_gpus

    def net(self, input):
        input = fluid.layers.reshape(input, shape=[-1, self.feat_dim])
        x = fluid.layers.fc(input=input, size=self.hidden_dim)
        x = fluid.layers.relu(0.1 * x)
        x = fluid.layers.fc(input=input, size=1)
        x = fluid.layers.sigmoid(0.1 * x)
        return x

    def loss_func(self, pred_score, gt_iou):
        gt_iou = fluid.layers.reshape(gt_iou, shape=[-1, 1])

        u_hmask = fluid.layers.cast(x=gt_iou > 0.6, dtype=DATATYPE)
        u_mmask = fluid.layers.logical_and(gt_iou <= 0.6, gt_iou > 0.2)
        u_mmask = fluid.layers.cast(x=u_mmask, dtype=DATATYPE)
        u_lmask = fluid.layers.logical_and(gt_iou <= 0.2, gt_iou >= 0.)
        u_lmask = fluid.layers.cast(x=u_lmask, dtype=DATATYPE)

        num_h = fluid.layers.cast(
            fluid.layers.reduce_sum(u_hmask), dtype=DATATYPE)
        num_m = fluid.layers.cast(
            fluid.layers.reduce_sum(u_mmask), dtype=DATATYPE)
        num_l = fluid.layers.cast(
            fluid.layers.reduce_sum(u_lmask), dtype=DATATYPE)

        r_m = num_h / num_m
        u_smmask = fluid.layers.uniform_random(
            shape=[self.mini_batch * self.top_K, 1],
            dtype=DATATYPE,
            min=0.0,
            max=1.0)
        u_smmask = fluid.layers.elementwise_mul(u_mmask, u_smmask)
        u_smmask = fluid.layers.cast(x=(u_smmask > (1. - r_m)), dtype=DATATYPE)

        r_l = 2 * num_h / num_l
        u_slmask = fluid.layers.uniform_random(
            shape=[self.mini_batch * self.top_K, 1],
            dtype=DATATYPE,
            min=0.0,
            max=1.0)
        u_slmask = fluid.layers.elementwise_mul(u_lmask, u_slmask)
        u_slmask = fluid.layers.cast(x=(u_slmask > (1. - r_l)), dtype=DATATYPE)

        weights = u_hmask + u_smmask + u_slmask
        weights.stop_gradient = True

        loss = fluid.layers.square_error_cost(pred_score, gt_iou)
        loss = fluid.layers.elementwise_mul(loss, weights)
        loss = 0.5 * fluid.layers.reduce_sum(loss) / fluid.layers.reduce_sum(
            weights)
        return [loss]
