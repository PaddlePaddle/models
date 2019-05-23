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
from __future__ import unicode_literals

import six

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay

from ..registry import YOLOHeads

__all__ = ['YOLOv3Head']


@YOLOHeads.register
class YOLOv3Head(object):
    """
    YOLOv3Head class of YOLO head block for YOLOv3 network

    The naming rules are same as them in
    https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/yolov3/models/yolov3.py

    Args:
        cfg (AttrDict): All parameters in dictionary.

    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.bn_decay = getattr(cfg.OPTIMIZER.WEIGHT_DECAY, 'BN_DECAY', False)
        self.class_num = getattr(cfg.DATA, 'CLASS_NUM')
        self._get_and_check_anchors()

    def _conv_bn(self,
                input,
                ch_out,
                filter_size,
                stride,
                padding,
                act='leaky',
                is_test=True,
                name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name+".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        bn_param_attr = ParamAttr(regularizer=L2Decay(float(self.bn_decay)),
                                  name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(regularizer=L2Decay(float(self.bn_decay)),
                                 name=bn_name + '.offset')
        out = fluid.layers.batch_norm(
            input=conv,
            act=None,
            is_test=is_test,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

    def _detection_block(self, input, channel, is_test=True, name=None):
        assert channel % 2 == 0, \
                "channel {} cannot be divided by 2 in \
                detection block {}".format(channel, name)

        conv = input
        for j in range(2):
            conv = self._conv_bn(conv, channel, filter_size=1, 
                                 stride=1, padding=0, is_test=is_test, 
                                 name='{}.{}.0'.format(name, j))
            conv = self._conv_bn(conv, channel*2, filter_size=3, 
                                 stride=1, padding=1, is_test=is_test, 
                                 name='{}.{}.1'.format(name, j))
        route = self._conv_bn(conv, channel, filter_size=1, stride=1, 
                              padding=0, is_test=is_test, 
                              name='{}.2'.format(name))
        tip = self._conv_bn(route,channel*2, filter_size=3, stride=1, 
                            padding=1, is_test=is_test, 
                            name='{}.tip'.format(name))
        return route, tip

    def _upsample(self, input, scale=2, name=None):
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

    def _get_and_check_anchors(self):
        """
        Check ANCHORS/ANCHOR_MASKS in config and parse mask_anchors

        """
        self.anchor_masks = getattr(self.cfg.YOLO_HEAD, 'ANCHOR_MASKS', [])
        anchors = getattr(self.cfg.YOLO_HEAD, 'ANCHORS', [])

        self.anchors = []
        self.mask_anchors = []

        assert len(anchors) > 0, "ANCHORS not set."
        assert len(self.anchor_masks) > 0, "ANCHOR_MASKS not set."

        for anchor in anchors:
            assert len(anchor) == 2, "anchor {} len should be 2".format(anchor)
            self.anchors.extend(anchor)

        anchor_num = len(anchors)
        for masks in self.anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def _get_outputs(self, inputs, is_train=True):
        """
        Get YOLOv3 head output

        Args:
            inputs ([Variable, ...]): Last Variable of each stage in backbone
            is_train (bool, default True): whether in train or test mode

        Returns:
            outputs ([Variable, ...]): Variables of each output layer

        """

        outputs = []

        # get last out_layer_num blocks in reverse order
        out_layer_num = len(self.anchor_masks)
        blocks = inputs[-1: -out_layer_num-1: -1]

        for i, block in enumerate(blocks):
            if i > 0: # perform concat in first 2 detection_block
                block = fluid.layers.concat(
                    input=[route, block],
                    axis=1)
            route, tip = self._detection_block(block, 
                                        channel=512//(2**i), 
                                        is_test=(not is_train),
                                        name="yolo_block.{}".format(i))

            # out channel number = mask_num * (5 + class_num)
            num_filters = len(self.anchor_masks[i]) * (self.class_num + 5)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(name="yolo_output.{}.conv.weights".format(i)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.),
                                    name="yolo_output.{}.conv.bias".format(i)))
            outputs.append(block_out)

            if i < len(blocks) - 1: 
                # do not perform upsample in the last detection_block
                route = self._conv_bn(input=route,
                                      ch_out=256//(2**i),
                                      filter_size=1,
                                      stride=1,
                                      padding=0,
                                      is_test=(not is_train),
                                      name="yolo_transition.{}".format(i))
                # upsample
                route = self._upsample(route)
        
        return outputs

    def get_loss(self, inputs, gt_box, gt_label, gt_score):
        """
        Get final loss of network of YOLOv3.

        Args:
            inputs ([Variable, ...]): Last Variable of each stage in backbone.
            gt_box (Variable): The ground-truth boudding boxes.
            gt_label (Variable): The ground-truth class labels.
            gt_score (Variable): The ground-truth boudding boxes mixup scores.

        Returns: 
            loss (Variable): The loss Variable of YOLOv3 network.
            
        """
        outputs = self._get_outputs(inputs, is_train=True)

        ignore_thresh = getattr(self.cfg.YOLO_HEAD, 'IGNORE_THRESH', 0.7)
        label_smooth = getattr(self.cfg.YOLO_HEAD, 'LABEL_SMOOTH', True)

        losses = []
        downsample = 32
        for i, output in enumerate(outputs):
            anchor_mask = self.anchor_masks[i]
            loss = fluid.layers.yolov3_loss(
                    x=output,
                    gt_box=gt_box,
                    gt_label=gt_label,
                    gt_score=gt_score,
                    anchors=self.anchors,
                    anchor_mask=anchor_mask,
                    class_num=self.class_num,
                    ignore_thresh=ignore_thresh,
                    downsample_ratio=downsample,
                    use_label_smooth=label_smooth,
                    name="yolo_loss"+str(i))
            losses.append(fluid.layers.reduce_mean(loss))
            downsample //= 2

        return sum(losses)

    def get_prediction(self, inputs, im_shape):
        """
        Get prediction result of YOLOv3 network

        Args:
            inputs ([Variable, ...]): Last Variable of each stage in backbone.
            im_shape (Variable): Variable of shape([h, w]) of each image

        Returns:
            pred (Variable): The prediction result after non-max suppress.

        """

        outputs = self._get_outputs(inputs, is_train=False)

        valid_thresh = getattr(self.cfg.TEST, 'VALID_THRESH', 0.01)
        nms_topk = getattr(self.cfg.TEST, 'NMS_TOPK', 400)
        nms_posk = getattr(self.cfg.TEST, 'NMS_POSK', 100)
        nms_thresh = getattr(self.cfg.TEST, 'NMS_THRESH', 0.45)

        boxes = []
        scores = []
        downsample = 32
        for i, output in enumerate(outputs):
            box, score= fluid.layers.yolo_box(x=output,
                                              img_size=im_shape,
                                              anchors=self.mask_anchors[i],
                                              class_num=self.class_num,
                                              conf_thresh=valid_thresh,
                                              downsample_ratio=downsample,
                                              name="yolo_box"+str(i))
            boxes.append(box)
            scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))

            downsample //= 2

        yolo_boxes = fluid.layers.concat(boxes, axis=1)
        yolo_scores = fluid.layers.concat(scores, axis=2)
        pred = fluid.layers.multiclass_nms(
                bboxes=yolo_boxes,
                scores=yolo_scores,
                score_threshold=valid_thresh,
                nms_top_k=nms_topk,
                keep_top_k=nms_posk,
                nms_threshold=nms_thresh,
                background_label=-1,
                name="multiclass_nms")
        return pred

