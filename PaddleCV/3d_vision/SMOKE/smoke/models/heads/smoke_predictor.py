
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from smoke.models.layers import group_norm, sigmoid_hm
from smoke.cvlibs import manager, param_init

@manager.HEADS.add_component
class SMOKEPredictor(nn.Layer):
    """SMOKE Predictor
    """
    def __init__(self,
                 num_classes=3,
                 reg_heads=10,
                 reg_channels=(1, 2, 3, 2, 2),
                 num_chanels=256,
                 norm_type="gn",
                 in_channels=64):
        super().__init__()
        
        regression = reg_heads
        regression_channels = reg_channels
        head_conv = num_chanels
        norm_func = nn.BatchNorm2D if norm_type == "bn" else group_norm
       
        assert sum(regression_channels) == regression, \
            "the sum of {} must be equal to regression channel of {}".format(
                reg_channels, reg_heads
            )

        self.dim_channel = get_channel_spec(regression_channels, name="dim")
        self.ori_channel = get_channel_spec(regression_channels, name="ori")


        self.class_head = nn.Sequential(
            nn.Conv2D(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias_attr=True),

            norm_func(head_conv),

            nn.ReLU(),

            nn.Conv2D(head_conv,
                      num_classes,
                      kernel_size=1,
                      padding=1 // 2,
                      bias_attr=True)
        )

        # todo: what is datafill here
        #self.class_head[-1].bias.data.fill_(-2.19)
        param_init.constant_init(self.class_head[-1].bias, value=-2.19)

        self.regression_head = nn.Sequential(
            nn.Conv2D(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias_attr=True),

            norm_func(head_conv),

            nn.ReLU(),

            nn.Conv2D(head_conv,
                      regression,
                      kernel_size=1,
                      padding=1 // 2,
                      bias_attr=True)
        )

        #_fill_fc_weights(self.regression_head)
        self.init_weight(self.regression_head)

    def forward(self, features):
        """predictor forward

        Args:
            features (paddle.Tensor): smoke backbone output

        Returns:
            list: sigmoid class heatmap and regression map
        """
        head_class = self.class_head(features)
        head_regression = self.regression_head(features)
        head_class = sigmoid_hm(head_class)

        # (N, C, H, W)

        # left slice bug
        # offset_dims = head_regression[:, self.dim_channel, :, :].clone()
        # head_regression[:, self.dim_channel, :, :] = F.sigmoid(offset_dims) - 0.5
        # vector_ori = head_regression[:, self.ori_channel, :, :].clone()
        # head_regression[:, self.ori_channel, :, :] = F.normalize(vector_ori)

        offset_dims = head_regression[:, self.dim_channel, :, :].clone()
        head_reg_dim = F.sigmoid(offset_dims) - 0.5
        
        vector_ori = head_regression[:, self.ori_channel, :, :].clone()
        head_reg_ori = F.normalize(vector_ori)

        head_regression_left = head_regression[:, :self.dim_channel.start, :, :]
        head_regression_right = head_regression[:, self.ori_channel.stop:, :, :]
        head_regression = paddle.concat([head_regression_left, head_reg_dim, head_reg_ori, head_regression_right], axis=1)

        
        return [head_class, head_regression]

    def init_weight(self, block):
        for sublayer in block.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                param_init.constant_init(sublayer.bias, value=0.0)


def get_channel_spec(reg_channels, name):
    """get dim and ori dim

    Args:
        reg_channels (tuple): regress channels, default(1, 2, 3, 2) for
        (depth_offset, keypoint_offset, dims, ori)
        name (str): dim or ori

    Returns:
        slice: for start channel to stop channel
    """
    if name == "dim":
        s = sum(reg_channels[:2])
        e = sum(reg_channels[:3])
    elif name == "ori":
        s = sum(reg_channels[:3])
        e = sum(reg_channels[:4])

    return slice(s, e, 1)