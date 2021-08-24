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


class SEBasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes,
                               planes,
                               kernel_size=3,
                               padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(nn.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 reduction=8):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes,
                               planes * 4,
                               kernel_size=1,
                               bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
                                nn.ReLU(),
                                nn.Linear(channel // reduction, channel),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape((b, c))
        y = self.fc(y).reshape((b, c, 1, 1))
        return x * y
