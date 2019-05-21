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

from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay

from config import cfg

from .darknet import add_DarkNet53_conv_body
from .darknet import conv_bn_layer

def yolo_detection_block(input, channel, is_test=True, name=None):
    assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)
    conv = input
    for j in range(2):
        conv = conv_bn_layer(conv, channel, filter_size=1, 
                             stride=1, padding=0, is_test=is_test, 
                             name='{}.{}.0'.format(name, j))
        conv = conv_bn_layer(conv, channel*2, filter_size=3, 
                             stride=1, padding=1, is_test=is_test, 
                             name='{}.{}.1'.format(name, j))
    route = conv_bn_layer(conv, channel, filter_size=1, stride=1, 
                          padding=0, is_test=is_test, 
                          name='{}.2'.format(name))
    tip = conv_bn_layer(route,channel*2, filter_size=3, stride=1, 
                        padding=1, is_test=is_test, 
                        name='{}.tip'.format(name))
    return route, tip

def upsample(input, scale=2,name=None):
    # get dynamic upsample output shape
    shape_nchw = fluid.layers.shape(input)
    shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
    shape_hw.stop_gradient = True
    in_shape = fluid.layers.cast(shape_hw, dtype='int32')
    out_shape = in_shape * scale
    out_shape.stop_gradient = True

    # reisze by actual_shape
    out = fluid.layers.resize_nearest(
        input=input,
        scale=scale,
        actual_shape=out_shape,
        name=name)
    return out

class YOLOv3(object):
    def __init__(self, 
                is_train=True,
                use_random=True):
        self.is_train = is_train
        self.use_random = use_random
        self.outputs = []
        self.losses = []
        self.downsample = 32

    def build_input(self):
        self.image_shape = [3, cfg.input_size, cfg.input_size]
        if self.is_train:
            self.py_reader = fluid.layers.py_reader(
                capacity=64,
                shapes = [[-1] + self.image_shape, 
                          [-1, cfg.max_box_num, 4], 
                          [-1, cfg.max_box_num], 
                          [-1, cfg.max_box_num]],
                lod_levels=[0, 0, 0, 0],
                dtypes=['float32'] * 2 + ['int32'] + ['float32'],
                use_double_buffer=True)
            self.image, self.gtbox, self.gtlabel, self.gtscore = \
                    fluid.layers.read_file(self.py_reader)
        else:
            self.image = fluid.layers.data(
                    name='image', shape=self.image_shape, dtype='float32'
                    )
            self.im_shape = fluid.layers.data(
                    name="im_shape", shape=[2], dtype='int32')
            self.im_id = fluid.layers.data(
                    name="im_id", shape=[1], dtype='int32')
    
    def feeds(self):
        if not self.is_train:
            return [self.image, self.im_id, self.im_shape]
        return [self.image, self.gtbox, self.gtlabel, self.gtscore]

    def build_model(self):
        self.build_input()

        self.outputs = []
        self.boxes = []
        self.scores = []

        blocks = add_DarkNet53_conv_body(self.image, not self.is_train)
        for i, block in enumerate(blocks):
            if i > 0:
                block = fluid.layers.concat(
                    input=[route, block],
                    axis=1)
            route, tip = yolo_detection_block(block, channel=512//(2**i), 
                                        is_test=(not self.is_train),
                                        name="yolo_block.{}".format(i))

            # out channel number = mask_num * (5 + class_num)
            num_filters = len(cfg.anchor_masks[i]) * (cfg.class_num + 5)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02),
                     name="yolo_output.{}.conv.weights".format(i)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0),
                                     regularizer=L2Decay(0.),
                                     name="yolo_output.{}.conv.bias".format(i)))
            self.outputs.append(block_out)

            if i < len(blocks) - 1:
                route = conv_bn_layer(
                    input=route,
                    ch_out=256//(2**i),
                    filter_size=1,
                    stride=1,
                    padding=0,
                    is_test=(not self.is_train),
                    name="yolo_transition.{}".format(i))
                # upsample
                route = upsample(route)


        for i, out in enumerate(self.outputs):
            anchor_mask = cfg.anchor_masks[i]

            if self.is_train:
                loss = fluid.layers.yolov3_loss(
                        x=out,
                        gt_box=self.gtbox,
                        gt_label=self.gtlabel,
                        gt_score=self.gtscore,
                        anchors=cfg.anchors,
                        anchor_mask=anchor_mask,
                        class_num=cfg.class_num,
                        ignore_thresh=cfg.ignore_thresh,
                        downsample_ratio=self.downsample,
                        use_label_smooth=cfg.label_smooth,
                        name="yolo_loss"+str(i))
                self.losses.append(fluid.layers.reduce_mean(loss))
            else:
                mask_anchors=[]
                for m in anchor_mask:
                    mask_anchors.append(cfg.anchors[2 * m])
                    mask_anchors.append(cfg.anchors[2 * m + 1])
                boxes, scores = fluid.layers.yolo_box(
                        x=out,
                        img_size=self.im_shape,
                        anchors=mask_anchors,
                        class_num=cfg.class_num,
                        conf_thresh=cfg.valid_thresh,
                        downsample_ratio=self.downsample,
                        name="yolo_box"+str(i))
                self.boxes.append(boxes)
                self.scores.append(fluid.layers.transpose(scores, perm=[0, 2, 1]))
                
            self.downsample //= 2


    def loss(self):
        return sum(self.losses)

    def get_pred(self):
        yolo_boxes = fluid.layers.concat(self.boxes, axis=1)
        yolo_scores = fluid.layers.concat(self.scores, axis=2)
        return fluid.layers.multiclass_nms(
                bboxes=yolo_boxes,
                scores=yolo_scores,
                score_threshold=cfg.valid_thresh,
                nms_top_k=cfg.nms_topk,
                keep_top_k=cfg.nms_posk,
                nms_threshold=cfg.nms_thresh,
                background_label=-1,
                name="multiclass_nms")

