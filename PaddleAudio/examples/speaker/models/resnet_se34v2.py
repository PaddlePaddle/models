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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleaudio
from paddleaudio.transforms import LogMelSpectrogram, MelSpectrogram

from .resnet_blocks import SEBasicBlock, SEBottleneck


class ResNetSE(nn.Layer):
    def __init__(self,
                 block,
                 layers,
                 num_filters,
                 feature_dim,
                 encoder_type='SAP',
                 n_mels=40,
                 log_input=True,
                 **kwargs):
        super(ResNetSE, self).__init__()

        print('Embedding size is %d, encoder %s.' % (feature_dim, encoder_type))

        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels = n_mels
        self.log_input = log_input

        self.conv1 = nn.Conv2D(1,
                               num_filters[0],
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2D(num_filters[0])

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block,
                                       num_filters[1],
                                       layers[1],
                                       stride=(2, 2))
        self.layer3 = self._make_layer(block,
                                       num_filters[2],
                                       layers[2],
                                       stride=(2, 2))
        self.layer4 = self._make_layer(block,
                                       num_filters[3],
                                       layers[3],
                                       stride=(2, 2))

        outmap_size = int(self.n_mels / 8)

        self.attention = nn.Sequential(
            nn.Conv1D(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1D(128),
            nn.Conv1D(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(axis=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, feature_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):

        out = paddle.create_parameter(size, 'float32')
        nn.initializer.XavierNormal(out)
        return out

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape((x.shape[0], -1, x.shape[-1]))
        w = self.attention(x)
        if self.encoder_type == "SAP":
            x = paddle.sum(x * w, axis=2)
        elif self.encoder_type == "ASP":
            mu = paddle.sum(x * w, axis=2)
            sg = paddle.sum((x**2) * w, axis=2) - mu**2
            sg = paddle.clip(sg, min=1e-5)
            sg = paddle.sqrt(sg)
            x = paddle.concat((mu, sg), 1)

        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)

        return x


def ResNetSE34V2(feature_dim=256, scale_factor=1, **kwargs):
    # Number of filters
    num_filters = [
        32 * scale_factor, 64 * scale_factor, 128 * scale_factor,
        256 * scale_factor
    ]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, feature_dim,
                     **kwargs)
    return model


if __name__ == '__main__':
    print(ResNetSE34V2())
