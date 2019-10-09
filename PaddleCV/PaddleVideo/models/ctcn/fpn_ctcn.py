#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#coding=UTF-8

import paddle.fluid as fluid
from paddle.fluid import ParamAttr
import numpy as np
from .ctcn_utils import get_ctcn_conv_initializer as get_init

DATATYPE = 'float32'


class FPNCTCN(object):
    def __init__(self, num_anchors, concept_size, num_classes, mode='train'):
        self.num_anchors = num_anchors
        self.concept_size = concept_size
        self.num_classes = num_classes
        self.is_training = (mode == 'train')

    def conv_bn_layer(self,
                      input,
                      ch_out,
                      filter_size,
                      stride=1,
                      padding=0,
                      act='relu'):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(initializer=get_init(input, filter_size)),
            bias_attr=False)
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            is_test=(not self.is_training), )

    def shortcut(self, input, planes, stride):
        if (input.shape[1] == planes * 4) and (stride == 1):
            return input
        else:
            return self.conv_bn_layer(input, planes * 4, 1, stride, act=None)

    def bottleneck_block(self, input, planes, stride=1):
        conv0 = self.conv_bn_layer(input, planes, filter_size=1)
        conv1 = self.conv_bn_layer(
            conv0, planes, filter_size=(3, 1), stride=stride, padding=(1, 0))
        conv2 = self.conv_bn_layer(conv1, planes * 4, filter_size=1, act=None)
        short = self.shortcut(input, planes, stride)
        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')

    def layer_warp(self, input, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            input = self.bottleneck_block(input, planes, stride)
        return input

    def upsample_add(self, x, y):
        _, _, H, W = y.shape
        upsample = fluid.layers.image_resize(
            x, out_shape=[H, W], resample='BILINEAR')
        return upsample + y

    def extractor(self, input):
        num_blocks = [3, 4, 6, 3]

        c1 = self.conv_bn_layer(
            input, ch_out=32, filter_size=(7, 1), stride=(2, 1), padding=(3, 0))

        c1 = self.conv_bn_layer(
            c1, ch_out=64, filter_size=(7, 1), stride=(2, 1), padding=(3, 0))

        c2 = self.layer_warp(c1, 64, num_blocks[0], 1)
        c3 = self.layer_warp(c2, 128, num_blocks[1], (2, 1))
        c4 = self.layer_warp(c3, 256, num_blocks[2], (2, 1))
        c5 = self.layer_warp(c4, 512, num_blocks[3], (2, 1))

        #feature pyramid
        p6 = fluid.layers.conv2d(
            c5,
            num_filters=512,
            filter_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            param_attr=ParamAttr(initializer=get_init(c5, (3, 1))))

        p7 = fluid.layers.relu(p6)
        p7 = fluid.layers.conv2d(
            p7,
            num_filters=512,
            filter_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            param_attr=ParamAttr(initializer=get_init(p7, (3, 1))))

        p8 = fluid.layers.relu(p7)
        p8 = fluid.layers.conv2d(
            p8,
            num_filters=512,
            filter_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            param_attr=ParamAttr(initializer=get_init(p8, (3, 1))))

        p9 = fluid.layers.relu(p8)
        p9 = fluid.layers.conv2d(
            p9,
            num_filters=512,
            filter_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            param_attr=ParamAttr(initializer=get_init(p9, (3, 1))))

        #top_down
        p5 = fluid.layers.conv2d(
            c5,
            512,
            1,
            1,
            0,
            param_attr=ParamAttr(initializer=get_init(c5, 1)), )

        p4 = self.upsample_add(
            p5,
            fluid.layers.conv2d(
                c4,
                512,
                1,
                1,
                0,
                param_attr=ParamAttr(initializer=get_init(c4, 1)), ))

        p3 = self.upsample_add(
            p4,
            fluid.layers.conv2d(
                c3,
                512,
                1,
                1,
                0,
                param_attr=ParamAttr(initializer=get_init(c3, 1)), ))

        p2 = self.upsample_add(
            p3,
            fluid.layers.conv2d(
                c2,
                512,
                1,
                1,
                0,
                param_attr=ParamAttr(initializer=get_init(c2, 1))))
        #smooth
        p4 = fluid.layers.conv2d(
            p4,
            num_filters=512,
            filter_size=(3, 1),
            stride=1,
            padding=(1, 0),
            param_attr=ParamAttr(initializer=get_init(p4, (3, 1))), )

        p3 = fluid.layers.conv2d(
            p3,
            num_filters=512,
            filter_size=(3, 1),
            stride=1,
            padding=(1, 0),
            param_attr=ParamAttr(initializer=get_init(p3, (3, 1))), )

        p2 = fluid.layers.conv2d(
            p2,
            num_filters=512,
            filter_size=(3, 1),
            stride=1,
            padding=(1, 0),
            param_attr=ParamAttr(initializer=get_init(p2, (3, 1))), )

        return p2, p3, p4, p5, p6, p7, p8, p9

    def net(self, input):
        fm_sizes = self.concept_size  # 402
        num_anchors = self.num_anchors  # 7

        loc_preds = []
        cls_preds = []
        # build fpn network
        xs = self.extractor(input)
        # build predict head
        for i, x in enumerate(xs):
            loc_pred = fluid.layers.dropout(
                x, dropout_prob=0.5, is_test=(not self.is_training))
            loc_pred = fluid.layers.conv2d(
                loc_pred,
                num_filters=256,
                filter_size=(3, 1),
                stride=1,
                padding=(1, 0),
                param_attr=ParamAttr(
                    name='loc_pred_conv1_weights',
                    initializer=get_init(loc_pred, (3, 1))),
                bias_attr=ParamAttr(
                    name='loc_pred_conv1_bias', ))

            loc_pred = fluid.layers.conv2d(
                loc_pred,
                num_filters=num_anchors * 2,
                filter_size=(1, fm_sizes),
                stride=1,
                padding=0,
                param_attr=ParamAttr(
                    name='loc_pred_conv2_weights',
                    initializer=get_init(loc_pred, (1, fm_sizes))),
                bias_attr=ParamAttr(
                    name='loc_pred_conv2_bias', ))

            loc_pred = 10.0 * fluid.layers.sigmoid(loc_pred) - 5.0
            loc_pred = fluid.layers.transpose(loc_pred, perm=[0, 2, 3, 1])
            tmp_size1 = loc_pred.shape[1] * loc_pred.shape[2] * loc_pred.shape[
                3] // 2
            loc_pred = fluid.layers.reshape(
                x=loc_pred, shape=[loc_pred.shape[0], tmp_size1, 2])
            loc_preds.append(loc_pred)

            cls_pred = fluid.layers.dropout(
                x, dropout_prob=0.5, is_test=(not self.is_training))
            cls_pred = fluid.layers.conv2d(
                cls_pred,
                num_filters=512,
                filter_size=(3, 1),
                stride=1,
                padding=(1, 0),
                param_attr=ParamAttr(
                    name='cls_pred_conv1_weights',
                    initializer=get_init(cls_pred, (3, 1))),
                bias_attr=ParamAttr(
                    name='cls_pred_conv1_bias', ))

            cls_pred = fluid.layers.conv2d(
                cls_pred,
                num_filters=num_anchors * self.num_classes,
                filter_size=(1, fm_sizes),
                stride=1,
                padding=0,
                param_attr=ParamAttr(
                    name='cls_pred_conv2_weights',
                    initializer=get_init(cls_pred, (1, fm_sizes))),
                bias_attr=ParamAttr(
                    name='cls_pred_conv2_bias', ))

            cls_pred = fluid.layers.transpose(cls_pred, perm=[0, 2, 3, 1])
            tmp_size2 = cls_pred.shape[1] * cls_pred.shape[2] * cls_pred.shape[
                3] // self.num_classes
            cls_pred = fluid.layers.reshape(
                x=cls_pred,
                shape=[cls_pred.shape[0], tmp_size2, self.num_classes])
            cls_preds.append(cls_pred)

        loc_preds = fluid.layers.concat(input=loc_preds, axis=1)
        cls_preds = fluid.layers.concat(input=cls_preds, axis=1)
        return loc_preds, cls_preds

    def hard_negative_mining(self, cls_loss, pos_bool):
        pos = fluid.layers.cast(pos_bool, dtype=DATATYPE)
        cls_loss = cls_loss * (pos - 1)
        _, indices = fluid.layers.argsort(cls_loss, axis=1)
        indices = fluid.layers.cast(indices, dtype=DATATYPE)
        _, rank = fluid.layers.argsort(indices, axis=1)

        num_neg = 3 * fluid.layers.reduce_sum(pos, dim=1)
        num_neg = fluid.layers.reshape(x=num_neg, shape=[-1, 1])
        neg = rank < num_neg
        return neg

    def loss(self, loc_preds, cls_preds, loc_targets, cls_targets):
        """
        param loc_targets: [N, 1785,2]
        param cls_targets: [N, 1785]
        """

        loc_targets.stop_gradient = True
        cls_targets.stop_gradient = True

        pos = cls_targets > 0
        pos_bool = pos
        pos = fluid.layers.cast(pos, dtype=DATATYPE)
        num_pos = fluid.layers.reduce_sum(pos)
        pos = fluid.layers.unsqueeze(pos, axes=[2])
        mask = fluid.layers.expand(pos, expand_times=[1, 1, 2])
        mask.stop_gradient = True

        loc_loss = fluid.layers.smooth_l1(
            loc_preds, loc_targets, inside_weight=mask, outside_weight=mask)
        loc_loss = fluid.layers.reduce_sum(loc_loss)

        cls_loss = fluid.layers.softmax_with_cross_entropy(
            logits=fluid.layers.reshape(
                cls_preds, shape=[-1, self.num_classes]),
            label=fluid.layers.reshape(
                cls_targets, shape=[-1, 1]),
            numeric_stable_mode=True)

        cls_loss = fluid.layers.reshape(
            cls_loss, shape=[-1, loc_targets.shape[1]])
        not_ignore = cls_targets >= 0
        not_ignore = fluid.layers.cast(not_ignore, dtype=DATATYPE)
        not_ignore.stop_gradient = True

        cls_loss = cls_loss * not_ignore

        neg = self.hard_negative_mining(cls_loss, pos_bool)
        neg = fluid.layers.cast(neg, dtype='bool')
        pos_bool = fluid.layers.cast(pos_bool, dtype='bool')

        selects = fluid.layers.logical_or(pos_bool, neg)
        selects = fluid.layers.cast(selects, dtype=DATATYPE)
        selects.stop_gradient = True
        cls_loss = cls_loss * selects
        cls_loss = fluid.layers.reduce_sum(cls_loss)
        alpha = 2.0
        loss = (alpha * loc_loss + cls_loss) / num_pos
        num_pos.stop_gradient = True
        return loss, alpha * loc_loss / num_pos, cls_loss / num_pos
