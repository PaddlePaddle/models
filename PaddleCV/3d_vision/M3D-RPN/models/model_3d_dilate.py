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
"""
model_3d_dilate
"""

from lib.rpn_util import *
from models.backbone.densenet import densenet121

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.container import Sequential
from paddle.fluid.initializer import Normal
import math


def initial_type(name,
                 input_channels,
                 init="kaiming",
                 use_bias=False,
                 filter_size=0,
                 stddev=0.02):
    if init == "kaiming":
        fan_in = input_channels * filter_size * filter_size
        bound = 1 / math.sqrt(fan_in)
        param_attr = fluid.ParamAttr(
            name=name + "_weight",
            initializer=fluid.initializer.Uniform(
                low=-bound, high=bound))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + '_bias',
                initializer=fluid.initializer.Uniform(
                    low=-bound, high=bound))
        else:
            bias_attr = False
    else:
        param_attr = fluid.ParamAttr(
            name=name + "_weight",
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=stddev))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + "_bias",
                initializer=fluid.initializer.Constant(0.0))
        else:
            bias_attr = False
    return param_attr, bias_attr


class ConvLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 padding=0,
                 stride=1,
                 groups=None,
                 act=None,
                 name=None):
        super(ConvLayer, self).__init__()

        param_attr, bias_attr = initial_type(
            name=name,
            input_channels=num_channels,
            use_bias=True,
            filter_size=filter_size)

        self.num_filters = num_filters
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            padding=padding,
            stride=stride,
            groups=groups,
            act=act,
            param_attr=param_attr,
            bias_attr=bias_attr)

    def forward(self, inputs):
        x = self._conv(inputs)
        return x


class RPN(fluid.dygraph.Layer):
    def __init__(self, phase, base, conf):
        super(RPN, self).__init__()
        self.base = base

        del self.base.transition3.pool

        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]
        self.prop_feats = ConvLayer(
            num_channels=self.base.num_features,
            num_filters=512,
            filter_size=3,
            padding=1,
            act='relu',
            name='rpn_prop_feats')
        self.cls = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_classes * self.num_anchors,
            filter_size=1,
            name='rpn_cls')

        self.bbox_x = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_anchors,
            filter_size=1,
            name='rpn_bbox_x')

        self.bbox_y = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_anchors,
            filter_size=1,
            name='rpn_bbox_y')

        self.bbox_w = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_anchors,
            filter_size=1,
            name='rpn_bbox_w')

        self.bbox_h = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_anchors,
            filter_size=1,
            name='rpn_bbox_h')

        self.bbox_x3d = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_anchors,
            filter_size=1,
            name='rpn_bbox_x3d')

        self.bbox_y3d = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_anchors,
            filter_size=1,
            name='rpn_bbox_y3d')

        self.bbox_z3d = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_anchors,
            filter_size=1,
            name='rpn_bbox_z3d')

        self.bbox_w3d = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_anchors,
            filter_size=1,
            name='rpn_bbox_w3d')

        self.bbox_h3d = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_anchors,
            filter_size=1,
            name='rpn_bbox_h3d')

        self.bbox_l3d = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_anchors,
            filter_size=1,
            name='rpn_bbox_l3d')

        self.bbox_rY3d = ConvLayer(
            num_channels=self.prop_feats.num_filters,
            num_filters=self.num_anchors,
            filter_size=1,
            name='rpn_bbox_rY3d')

        self.feat_stride = conf.feat_stride

        self.feat_size = calc_output_size(
            np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size,
                                   conf.feat_stride)
        self.anchors = conf.anchors

    def forward(self, inputs):
        # backbone

        x = self.base(inputs)
        prop_feats = self.prop_feats(x)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)

        batch_size, c, feat_h, feat_w = cls.shape
        feat_size = fluid.layers.shape(cls)[2:4]

        # reshape for cross entropy
        cls = fluid.layers.reshape(
            x=cls,
            shape=[
                batch_size, self.num_classes, feat_h * self.num_anchors, feat_w
            ])
        # score probabilities
        prob = fluid.layers.softmax(cls, axis=1)

        # reshape for consistency
        bbox_x = flatten_tensor(
            fluid.layers.reshape(
                x=bbox_x,
                shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_y = flatten_tensor(
            fluid.layers.reshape(
                x=bbox_y,
                shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_w = flatten_tensor(
            fluid.layers.reshape(
                x=bbox_w,
                shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_h = flatten_tensor(
            fluid.layers.reshape(
                x=bbox_h,
                shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))

        bbox_x3d = flatten_tensor(
            fluid.layers.reshape(
                x=bbox_x3d,
                shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_y3d = flatten_tensor(
            fluid.layers.reshape(
                x=bbox_y3d,
                shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_z3d = flatten_tensor(
            fluid.layers.reshape(
                x=bbox_z3d,
                shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_w3d = flatten_tensor(
            fluid.layers.reshape(
                x=bbox_w3d,
                shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_h3d = flatten_tensor(
            fluid.layers.reshape(
                x=bbox_h3d,
                shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_l3d = flatten_tensor(
            fluid.layers.reshape(
                x=bbox_l3d,
                shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_rY3d = flatten_tensor(
            fluid.layers.reshape(
                x=bbox_rY3d,
                shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))

        # bundle
        bbox_2d = fluid.layers.concat(
            input=[bbox_x, bbox_y, bbox_w, bbox_h], axis=2)
        bbox_3d = fluid.layers.concat(
            input=[
                bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d,
                bbox_rY3d
            ],
            axis=2)

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.phase == "train":
            return cls, prob, bbox_2d, bbox_3d, feat_size

        else:
            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                #self.feat_size = [feat_h, feat_w]
                #self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride)
                self.rois = locate_anchors(self.anchors, [feat_h, feat_w],
                                           self.feat_stride)

        return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


def build(conf, backbone, phase='train'):

    train = phase.lower() == 'train'

    if backbone.lower() == "densenet121":
        model_backbone = densenet121()  # pretrain 

    rpn_net = RPN(phase, model_backbone.features, conf)

    # pretrain 
    if 'pretrained' in conf and conf.pretrained is not None:
        print("load pretrain model from ", conf.pretrained)
        pretrained, _ = fluid.load_dygraph(conf.pretrained)
        rpn_net.base.set_dict(pretrained, use_structured_name=True)

    if train: rpn_net.train()
    else: rpn_net.eval()

    return rpn_net
