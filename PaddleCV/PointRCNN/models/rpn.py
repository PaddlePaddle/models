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

from models.pointnet2_msg import PointNet2MSG
from models.pointnet2_modules import conv_bn
from models.loss_utils import sigmoid_focal_loss, get_reg_loss


__all__ = ["PointNet2SemSegSSG", "PointNet2SemSegMSG"]


class RPN(object):
    def __init__(self, cfg, use_xyz=True, mode='TRAIN'):
        self.cfg = cfg
        self.use_xyz = use_xyz
        self.mode = mode
        self.is_train = mode == 'TRAIN'
        self.inputs = None

    def build(self, inputs):
        assert self.cfg.RPN.BACKBONE == 'pointnet2_msg', \
                "RPN backbone only support pointnet2_msg"
        self.inputs = inputs

        xyz = inputs["pts_input"]
        assert not self.cfg.RPN.USE_INTENSITY, \
                "RPN.USE_INTENSITY not support now"
        feature = None
        msg = PointNet2MSG(self.cfg, xyz, feature, self.use_xyz)
        self.backbone_xyz, self.backbone_feature = msg.build()
        cls_out = fluid.layers.unsqueeze(self.backbone_feature, axes=[-1])
        reg_out = cls_out

        # classification branch
        for i in range(self.cfg.RPN.CLS_FC.__len__()):
            cls_out = conv_bn(cls_out, self.cfg.RPN.CLS_FC[i], bn=self.cfg.RPN.USE_BN, name='rpn_cls_{}'.format(i))
            # if i == 0 and self.cfg.RPN.DP_RATIO > 0:
            #     cls_out = fluid.layers.dropout(cls_out, self.cfg.RPN.DP_RATIO)
        cls_out = conv_bn(cls_out, 1, bn=False, act=None, name='rpn_cls_out')
        self.cls_out = fluid.layers.squeeze(cls_out, axes=[1, 3])

        # regression branch
        per_loc_bin_num = int(self.cfg.RPN.LOC_SCOPE / self.cfg.RPN.LOC_BIN_SIZE) * 2
        if self.cfg.RPN.LOC_XZ_FINE:
            reg_channel = per_loc_bin_num * 4 + self.cfg.RPN.NUM_HEAD_BIN * 2 + 3
        else:
            reg_channel = per_loc_bin_num * 2 + self.cfg.RPN.NUM_HEAD_BIN * 2 + 3
        reg_channel += 1  # reg y

        for i in range(self.cfg.RPN.REG_FC.__len__()):
            reg_out = conv_bn(reg_out, self.cfg.RPN.REG_FC[i], bn=self.cfg.RPN.USE_BN, name='rpn_reg_{}'.format(i))
            # if i == 0 and self.cfg.RPN.DP_RATIO > 0:
            #     reg_out = fluid.layers.dropout(reg_out, self.cfg.RPN.DP_RATIO)
        reg_out = conv_bn(reg_out, reg_channel, bn=False, act=None, name='rpn_reg_out')
        reg_out = fluid.layers.squeeze(reg_out, axes=[3])
        self.reg_out = fluid.layers.transpose(reg_out, [0, 2, 1])

    def get_outputs(self):
        return {
            "backbone_xyz": self.backbone_xyz,
            "backbone_feature": self.backbone_feature,
            "rpn_cls": self.cls_out,
            "rpn_reg": self.reg_out,
        }

    def get_loss(self):
        assert self.inputs is not None, \
                "please call build() first"
        rpn_cls_label = self.inputs['rpn_cls_label']
        rpn_reg_label = self.inputs['rpn_reg_label']

        # RPN classification loss
        assert self.cfg.RPN.LOSS_CLS == "SigmoidFocalLoss", \
                "unsupported RPN cls loss type {}".format(self.cfg.RPN.LOSS_CLS)
        cls_flat = fluid.layers.reshape(self.cls_out, shape=[-1])
        cls_label_flat = fluid.layers.reshape(rpn_cls_label, shape=[-1])
        pos = fluid.layers.cast(cls_label_flat > 0, dtype=cls_flat.dtype)
        pos_normalizer = fluid.layers.reduce_sum(pos)
        cls_weights = fluid.layers.cast(cls_label_flat >= 0, dtype=cls_flat.dtype)
        cls_weights = cls_weights / fluid.layers.clip(pos_normalizer, min=1.0, max=1e10)
        cls_label_flat = fluid.layers.cast(cls_label_flat, dtype=cls_flat.dtype)
        rpn_loss_cls = sigmoid_focal_loss(cls_flat, cls_label_flat, cls_weights)
        rpn_loss_cls = fluid.layers.reduce_sum(rpn_loss_cls)
        # fg_num = fluid.layers.cast(rpn_cls_label > 0., dtype='int32')
        # fg_num = fluid.layers.reduce_sum(fg_num, dim=[1])
        # rpn_loss_cls = fluid.layers.sigmoid_focal_loss(self.cls_out, rpn_cls_label, fg_num, 
        #             alpha=cfg.RPN.FOCAL_ALPHA[0], gamma=cfg.RPN.FOCAL_GAMMA)

        # RPN regression loss
        reg_out = fluid.layers.reshape(self.reg_out, [-1, self.reg_out.shape[-1]])
        reg_label = fluid.layers.reshape(rpn_reg_label, [-1, rpn_reg_label.shape[-1]])
        fg_mask = fluid.layers.cast(cls_label_flat > 0, dtype=reg_out.dtype)
        loc_loss, angle_loss, size_loss, loss_dict = get_reg_loss(
                                        reg_out, reg_label, fg_mask, 16384.,
                                        loc_scope=self.cfg.RPN.LOC_SCOPE,
                                        loc_bin_size=self.cfg.RPN.LOC_BIN_SIZE,
                                        num_head_bin=self.cfg.RPN.NUM_HEAD_BIN,
                                        anchor_size=self.cfg.CLS_MEAN_SIZE[0],
                                        get_xz_fine=self.cfg.RPN.LOC_XZ_FINE,
                                        get_y_by_bin=False,
                                        get_ry_fine=False)
        rpn_loss_reg = loc_loss + angle_loss + size_loss * 3

        rpn_loss = rpn_loss_cls * self.cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * self.cfg.RPN.LOSS_WEIGHT[1]
        return rpn_loss_cls, rpn_loss_reg
        
if __name__ == "__main__":
    from utils.config import load_config, cfg
    load_config('./cfgs/default.yml')
    cfg.RPN.ENABLED = True
    cfg.RCNN.ENABLED = False
    keys = ['pts_input', 'rpn_cls_label', 'rpn_reg_label', 'pts_rect', 'pts_features', 'gt_boxes3d']
    np_inputs = {}
    for key in keys:
        np_inputs[key] = np.expand_dims(np.load('/paddle/rpn_data/{}.npy'.format(key)), axis=0)
        print(key, np_inputs[key].shape)

    pts_input = fluid.layers.data(name='pts_input', shape=[16384, 3], dtype='float32')
    rpn_cls_label = fluid.layers.data(name='rpn_cls_label', shape=[16384], dtype='int32')
    rpn_reg_label = fluid.layers.data(name='rpn_reg_label', shape=[16384, 7], dtype='float32')
    inputs = {
            "pts_input": pts_input,
            "rpn_cls_label": rpn_cls_label,
            "rpn_reg_label": rpn_reg_label,
            }
    rpn = RPN(cfg)
    rpn.build(inputs)
    loss_cls, loss_reg = rpn.get_loss()

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    ret = exe.run(fetch_list=[
        loss_cls.name,
        loss_reg.name,
        # rpn.backbone_xyz.name,
        # rpn.backbone_feature.name,
        # rpn.cls_out.name,
        # rpn.reg_out.name,
        "reshape2_2.tmp_0",
        "reshape2_3.tmp_0",
        ], feed={'pts_input': np_inputs['pts_input'], 'rpn_cls_label': np_inputs['rpn_cls_label'], 'rpn_reg_label': np_inputs['rpn_reg_label']})
    print(ret)
