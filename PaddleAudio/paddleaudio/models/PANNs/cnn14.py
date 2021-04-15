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

import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...utils.log import logger
from .conv import ConvBlock


class CNN14(nn.Layer):
    """
    The CNN14(14-layer CNNs) mainly consist of 6 convolutional blocks while each convolutional
    block consists of 2 convolutional layers with a kernel size of 3 × 3.

    Reference:
        PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
        https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 2048

    def __init__(self, extract_embedding: bool = True, checkpoint: str = None):

        super(CNN14, self).__init__()
        self.bn0 = nn.BatchNorm2D(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)

        if checkpoint is not None and os.path.isfile(checkpoint):
            state_dict = paddle.load(checkpoint)
            self.set_state_dict(state_dict)
            logger.info(f'Loaded CNN14 pretrained parameters from: {checkpoint}')
        else:
            logger.error('No valid checkpoints for CNN14. Start training from scratch.')

        self.extract_embedding = extract_embedding

    def forward(self, x):
        x.stop_gradient = False
        x = x.transpose([0, 3, 2, 1])
        x = self.bn0(x)
        x = x.transpose([0, 3, 2, 1])

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.mean(axis=3)
        x = x.max(axis=2) + x.mean(axis=2)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))
        return output
