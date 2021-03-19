import os
from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np


class EnvNet2(nn.Layer):
    """
    EnvNet-V2
    https://openreview.net/forum?id=B1Gi6LeRZ
    """

    def __init__(self, n_class, checkpoint=None):
        super(EnvNet2, self).__init__()
        self.conv1 = ConvBNReLU(1, 32, (1, 64), (1, 2))
        self.conv2 = ConvBNReLU(32, 64, (1, 16), (1, 2))
        self.pool2 = nn.MaxPool2D(kernel_size=(1, 64), stride=(1, 64))

        self.conv3 = ConvBNReLU(1, 32, (8, 8))  # input after swap axes
        self.conv4 = ConvBNReLU(32, 32, (8, 8))
        self.pool4 = nn.MaxPool2D(kernel_size=(5, 3), stride=(5, 3))

        self.conv5 = ConvBNReLU(32, 64, (1, 4))
        self.conv6 = ConvBNReLU(64, 64, (1, 4))
        self.pool6 = nn.MaxPool2D(kernel_size=(1, 2), stride=(1, 2))

        self.conv7 = ConvBNReLU(64, 128, (1, 2))
        self.conv8 = ConvBNReLU(128, 128, (1, 2))
        self.pool8 = nn.MaxPool2D(kernel_size=(1, 2), stride=(1, 2))

        self.conv9 = ConvBNReLU(128, 256, (1, 2))
        self.conv10 = ConvBNReLU(256, 256, (1, 2))
        self.pool10 = nn.MaxPool2D(kernel_size=(1, 2), stride=(1, 2))
        self.flatten = nn.Flatten()

        self.fc1 = FCDN(256 * 10 * 8, 4096)  # input after flatten
        self.fc2 = FCDN(4096, 4096)
        self.output = FCDN(4096, n_class, None, 0)

        if checkpoint is not None and os.path.isfile(checkpoint):
            state_dict = paddle.load(checkpoint)
            self.set_state_dict(state_dict)
            print('Loaded model parameters from %s' %
                  os.path.abspath(checkpoint))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.pool2(y)
        y = paddle.transpose(y, perm=[0, 2, 1, 3])

        y = self.conv3(y)
        y = self.conv4(y)
        y = self.pool4(y)

        y = self.conv5(y)
        y = self.conv6(y)
        y = self.pool6(y)

        y = self.conv7(y)
        y = self.conv8(y)
        y = self.pool8(y)

        y = self.conv9(y)
        y = self.conv10(y)
        y = self.pool10(y)

        y = self.flatten(y)
        y = self.fc1(y)
        y = self.fc2(y)
        logits = self.output(y)

        return logits


class ConvBNReLU(nn.Layer):
    """
    Convolution network block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 use_bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            weight_attr=nn.initializer.KaimingUniform(),
            bias_attr=False, )
        self.batch_norm = nn.BatchNorm(num_channels=out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        layer = self.conv(x)
        layer = self.batch_norm(layer)
        layer = self.activation(layer)
        return layer


class FCDN(nn.Layer):
    """
    Full-connected network block.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 activation='relu',
                 dropout=0.5):
        super(FCDN, self).__init__()
        self.fcn = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            weight_attr=nn.initializer.KaimingUniform(), )
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'softmax':
            self.activation = nn.Softmax()
        else:
            self.activation = None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x):
        fc = self.fcn(x)
        fc = self.activation(fc) if self.activation is not None else fc
        fc = self.dropout(fc) if self.dropout is not None else fc
        return fc


if __name__ == "__main__":
    np.random.seed(0)

    n_class = 50
    batch_size = 5
    model = EnvNet2(n_class=n_class)

    inp = np.random.random(size=(batch_size, 1, 1, 66650))
    inp = paddle.to_tensor(inp, dtype='float32')
    model(inp)
