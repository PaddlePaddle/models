    
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

from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay

from config_parser import ConfigPaser
from config import cfg

from darknet import add_DarkNet53_conv_body
from darknet import conv_bn_layer

def yolo_detection_block(input, channel,i):
    assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
    conv1 = input
    for j in range(2):
        conv1 = conv_bn_layer(conv1, channel, filter_size=1, stride=1, padding=0,i=i+j*2)
        conv1 = conv_bn_layer(conv1, channel*2, filter_size=3, stride=1, padding=1,i=i+j*2+1)
    route = conv_bn_layer(conv1, channel, filter_size=1, stride=1, padding=0,i=i+4)
    tip = conv_bn_layer(route,channel*2, filter_size=3, stride=1, padding=1,i=i+5)
    return route, tip

def upsample(out, stride=2,name=None):
    out = out
    scale = stride
    # get dynamic upsample output shape
    shape_nchw = fluid.layers.shape(out)
    shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
    shape_hw.stop_gradient = True
    in_shape = fluid.layers.cast(shape_hw, dtype='int32')
    out_shape = in_shape * scale
    out_shape.stop_gradient = True

    # reisze by actual_shape
    out = fluid.layers.resize_nearest(
        input=out,
        scale=scale,
        actual_shape=out_shape,
        name=name)
    return out

class YOLOv3(object):
    def __init__(self, 
                model_cfg_path,
                is_train=True,
                use_pyreader=True,
                use_random=True):
        self.model_cfg_path = model_cfg_path
        self.config_parser = ConfigPaser(model_cfg_path)
        self.is_train = is_train
        self.use_pyreader = use_pyreader
        self.use_random = use_random
        self.outputs = []
        self.losses = []
        self.downsample = 32
        self.ignore_thresh = .7
        self.class_num = 80

    def build_model(self):

        self.img_height = cfg.input_size
        self.img_width = cfg.input_size

        self.build_input()

        out = self.image

        self.yolo_anchors = []
        self.yolo_classes = []
        self.outputs = []
        self.boxes = []
        self.scores = []


        scale1,scale2,scale3 = add_DarkNet53_conv_body(out)
         
        # 13*13 scale output
        route1, tip1 = yolo_detection_block(scale1, channel=512,i=75)
        # scale1 output
        scale1_out = fluid.layers.conv2d(
            input=tip1,
            num_filters=255,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02),
                 name="conv81_weights"),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0),
                                 regularizer=L2Decay(0.),
                                 name="conv81_bias"))

        self.outputs.append(scale1_out) 

        route1 = conv_bn_layer(
            input=route1,
            ch_out=256,
            filter_size=1,
            stride=1,
            padding=0,
            i=84)
        # upsample
        route1 = upsample(route1)

        # concat
        route1 = fluid.layers.concat(
            input=[route1,scale2],
            axis=1)

        # 26*26 scale output
        route2, tip2 = yolo_detection_block(route1, channel=256,i=87)
        
        # scale2 output
        scale2_out = fluid.layers.conv2d(
            input=tip2,
            num_filters=255,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(name="conv93_weights"),
            bias_attr=ParamAttr(name="conv93_bias"))

        self.outputs.append(scale2_out)

        route2 = conv_bn_layer(
            input=route2,
            ch_out=128,
            filter_size=1,
            stride=1,
            padding=0,
            i=96)
        # upsample
        route2 = upsample(route2)

        # concat
        route2 = fluid.layers.concat(
            input=[route2,scale3],
            axis=1)

        # 52*52 scale output
        route3, tip3 = yolo_detection_block(route2, channel=128, i=99)

        # scale3 output
        scale3_out = fluid.layers.conv2d(
            input=tip3,
            num_filters=255,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(name="conv105_weights"),
            bias_attr=ParamAttr(name="conv105_bias"))


        self.outputs.append(scale3_out)
        # yolo

        anchor_mask = [6,7,8,3,4,5,0,1,2]
        anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
        for i,out in enumerate(self.outputs):
            mask = anchor_mask[i*3 : (i+1)*3]
            mask_anchors=[]

            for m in mask:
                mask_anchors.append(anchors[2 * m])
                mask_anchors.append(anchors[2 * m + 1])
            self.yolo_anchors.append(mask_anchors)
            class_num = int(self.class_num)
            self.yolo_classes.append(class_num)

            if self.is_train:
                ignore_thresh = float(self.ignore_thresh)
                loss = fluid.layers.yolov3_loss(
                        x=out,
                        gtbox=self.gtbox,
                        gtlabel=self.gtlabel,
                        # gtscore=self.gtscore,
                        anchors=anchors,
                        anchor_mask=mask,
                        class_num=class_num,
                        ignore_thresh=ignore_thresh,
                        downsample_ratio=self.downsample,
                        # use_label_smooth=False,
                        name="yolo_loss"+str(i))
                self.losses.append(fluid.layers.reduce_mean(loss))
            else:
                boxes, scores = fluid.layers.yolo_box(
                        x=out,
                        img_size=self.im_shape,
                        anchors=mask_anchors,
                        class_num=class_num,
                        conf_thresh=cfg.valid_thresh,
                        downsample_ratio=self.downsample,
                        name="yolo_box"+str(i))
                self.boxes.append(boxes)
                self.scores.append(fluid.layers.transpose(scores, perm=[0, 2, 1]))
                
            self.downsample //= 2


    def loss(self):
        return sum(self.losses)

    def get_pred(self):
        # return self.outputs
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
    
    def get_yolo_anchors(self):
        return self.yolo_anchors

    def get_yolo_classes(self):
        return self.yolo_classes

    def build_input(self):
        self.image_shape = [3, self.img_height, self.img_width]
        if self.use_pyreader and self.is_train:
            self.py_reader = fluid.layers.py_reader(
                capacity=64,
                shapes = [[-1] + self.image_shape, [-1, cfg.max_box_num, 4], [-1, cfg.max_box_num], [-1, cfg.max_box_num]],
                lod_levels=[0, 0, 0, 0],
                dtypes=['float32'] * 2 + ['int32'] + ['float32'],
                use_double_buffer=True)
            self.image, self.gtbox, self.gtlabel, self.gtscore = fluid.layers.read_file(self.py_reader)
        else:
            self.image = fluid.layers.data(
                    name='image', shape=self.image_shape, dtype='float32'
                    )
            self.gtbox = fluid.layers.data(
                    name='gtbox', shape=[cfg.max_box_num, 4], dtype='float32'
                    )
            self.gtlabel = fluid.layers.data(
                    name='gtlabel', shape=[cfg.max_box_num], dtype='int32'
                    )
            self.gtscore = fluid.layers.data(
                    name='gtscore', shape=[cfg.max_box_num], dtype='float32'
                    )
            self.im_shape = fluid.layers.data(
                    name="im_shape", shape=[2], dtype='int32')
            self.im_id = fluid.layers.data(
                    name="im_id", shape=[1], dtype='int32')
    
    def feeds(self):
        if not self.is_train:
            return [self.image, self.im_id, self.im_shape]
        return [self.image, self.gtbox, self.gtlabel, self.gtscore]

    def get_input_size(self):
        return cfg.input_size


