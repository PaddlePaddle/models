# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2D(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias_attr=False)
        self.conv2 = nn.Conv2D(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.bn2 = nn.BatchNorm2D(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception(
                f'Pooling type of {pool_type} is not supported. It must be one of "max", "avg" and "avg+max".')
        return x


class ConvBlock5x5(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2D(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(2, 2),
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception(
                f'Pooling type of {pool_type} is not supported. It must be one of "max", "avg" and "avg+max".')
        return x
