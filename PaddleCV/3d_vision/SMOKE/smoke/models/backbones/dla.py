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
import numpy as np

from smoke.models.layers import group_norm
from smoke.cvlibs import manager
from smoke.utils import pretrained_utils

__all__ = [
    "DLA", "DLA34"
]

@manager.BACKBONES.add_component
class DLA(nn.Layer):
    
    def __init__(self, 
                 levels,
                 channels,
                 block,
                 down_ratio=4, 
                 last_level=5, 
                 out_channel=0,
                 norm_type="gn",
                 pretrained=None):

        super().__init__()

        self.pretrained = pretrained

        assert down_ratio in [2, 4, 8, 16]

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level

        norm_func = nn.BatchNorm2D if norm_type == "bn" else group_norm

        self.base = DLABase(levels, 
                            channels, 
                            block=eval(block), 
                            norm_func=norm_func)
        
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(startp=self.first_level,
                            channels=channels[self.first_level:],
                            scales=scales,
                            norm_func=norm_func)
        
        if out_channel == 0:
            out_channel = channels[self.first_level]

        up_scales = [2 ** i for i in range(self.last_level - self.first_level)]
        self.ida_up = IDAUp(in_channels=channels[self.first_level:self.last_level],
                            out_channel=out_channel,
                            up_f=up_scales,
                            norm_func=norm_func)
        self.init_weight()

    def forward(self, x):

        x = self.base(x)
        x = self.dla_up(x)
        y = []
        iter_levels = range(self.last_level - self.first_level)
        for i in iter_levels:

            y.append(x[i].clone())
        
        self.ida_up(y, 0, len(y))

        return y[-1]

    def init_weight(self):
       pretrained_utils.load_pretrained_model(self, self.pretrained)

class DLABase(nn.Layer):
    """DLA base module
    """
    def __init__(self,
                 levels,
                 channels,
                 block=None,
                 residual_root=False,
                 norm_func=None,
                 ):
        super().__init__()

        self.channels = channels
        self.level_length = len(levels)

        if block is None:
            block = BasicBlock
        if norm_func is None:
            norm_func = nn.BatchNorm2d       
        self.base_layer = nn.Sequential(
                            nn.Conv2D(3,
                                      channels[0],
                                      kernel_size=7,
                                      stride=1,
                                      padding=3,
                                      bias_attr=False),
                            norm_func(channels[0]),
                            nn.ReLU()
                        )

        self.level0 = _make_conv_level(in_channels=channels[0],
                                       out_channels=channels[0],
                                       num_convs=levels[0],
                                       norm_func=norm_func)

        self.level1 = _make_conv_level(in_channels=channels[0],
                                       out_channels=channels[1],
                                       num_convs=levels[0],
                                       norm_func=norm_func,
                                       stride=2)

        self.level2 = Tree(level=levels[2],
                           block=block,
                           in_channels=channels[1],
                           out_channels=channels[2],
                           norm_func=norm_func,
                           stride=2,
                           level_root=False,
                           root_residual=residual_root)

        self.level3 = Tree(level=levels[3],
                           block=block,
                           in_channels=channels[2],
                           out_channels=channels[3],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        self.level4 = Tree(level=levels[4],
                           block=block,
                           in_channels=channels[3],
                           out_channels=channels[4],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        self.level5 = Tree(level=levels[5],
                           block=block,
                           in_channels=channels[4],
                           out_channels=channels[5],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

    def forward(self, x):
        """forward
        """
        y = []
        x = self.base_layer(x)

        for i in range(self.level_length):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y

class DLAUp(nn.Layer):
    """DLA Up module
    """
    def __init__(self,
                 startp,
                 channels,
                 scales,
                 in_channels=None,
                 norm_func=None):
        """DLA Up module
        """
        super(DLAUp, self).__init__()

        self.startp = startp
        if norm_func is None:
            norm_func = nn.BatchNorm2d

        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)

        scales = np.array(scales, dtype=int)

        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self,
                    'ida_{}'.format(i),
                    IDAUp(in_channels[j:],
                          channels[j],
                          scales[j:] // scales[j],
                          norm_func))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        """forward
        """
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out

class BasicBlock(nn.Layer):
    """Basic Block
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_func,
                 stride=1,
                 dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2D(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias_attr=False,
                               dilation=dilation)
        self.norm1 = norm_func(out_channels)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2D(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=dilation,
                               bias_attr=False,
                               dilation=dilation
                               )
        self.norm2 = norm_func(out_channels)

    def forward(self, x, residual=None):
        """forward
        """
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)

        return out


class Tree(nn.Layer):

    def __init__(self,
                 level,
                 block,
                 in_channels,
                 out_channels,
                 norm_func,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False
                 ):
        super(Tree, self).__init__()

        if root_dim == 0:
            root_dim = 2 * out_channels

        if level_root:
            root_dim += in_channels

        if level == 1:
            self.tree1 = block(in_channels,
                               out_channels,
                               norm_func,
                               stride,
                               dilation=dilation)

            self.tree2 = block(out_channels,
                               out_channels,
                               norm_func,
                               stride=1,
                               dilation=dilation)
        else:
            new_level = level - 1
            self.tree1 = Tree(new_level,
                              block,
                              in_channels,
                              out_channels,
                              norm_func,
                              stride,
                              root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)

            self.tree2 = Tree(new_level,
                              block,
                              out_channels,
                              out_channels,
                              norm_func,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)
        if level == 1:
            self.root = Root(root_dim,
                             out_channels,
                             norm_func,
                             root_kernel_size,
                             root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.level = level

        self.downsample = None
        if stride > 1:
            self.downsample = nn.MaxPool2D(stride, stride=stride)

        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2D(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          bias_attr=False),

                norm_func(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        """forward
        """
        if children is None:
            children = []

        if self.downsample:
            bottom = self.downsample(x)
        else:
            bottom = x

        if self.project:
            residual = self.project(bottom)
        else:
            residual = bottom

        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)

        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class Root(nn.Layer):
    """Root module
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_func,
                 kernel_size,
                 residual):
        super(Root, self).__init__()

        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1,
                              bias_attr=False,
                              padding=(kernel_size - 1) // 2)

        self.norm = norm_func(out_channels)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, *x):
        """forward
        """
        children = x
        x = self.conv(paddle.concat(x, 1))
        x = self.norm(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x

class IDAUp(nn.Layer):
    """IDAUp module
    """
    def __init__(self,
                 in_channels,
                 out_channel,
                 up_f,  # todo: what is up_f here?
                 norm_func):
        super().__init__()

        for i in range(1, len(in_channels)):
            in_channel = in_channels[i]
            f = int(up_f[i])

            #USE_DEFORMABLE_CONV = False 
            
            # so far only support  normal convolution
            proj = NormalConv(in_channel, out_channel, norm_func)
            node = NormalConv(out_channel, out_channel, norm_func)

            up = nn.Conv2DTranspose(out_channel,
                                    out_channel,
                                    kernel_size=f * 2,
                                    stride=f,
                                    padding=f // 2,
                                    output_padding=0,
                                    groups=out_channel,
                                    bias_attr=False)
            # todo: uncommoment later
            # _fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        """forward
        """
        for i in range(startp + 1, endp):
            
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])

class NormalConv(nn.Layer):
    """Normal Conv without deformable
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_func):
        super(NormalConv, self).__init__()

        self.norm = norm_func(out_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size=(3, 3),
                              padding=1)

    def forward(self, x):
        """forward
        """
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x 

def _make_conv_level(in_channels, out_channels, num_convs, norm_func,
                     stride=1, dilation=1):
        """
        make conv layers based on its number.
        """
        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2D(in_channels, out_channels, kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation, bias_attr=False, dilation=dilation),
                norm_func(out_channels),
                nn.ReLU()])

            in_channels = out_channels

        return nn.Sequential(*layers)

@manager.BACKBONES.add_component
def DLA34(**kwargs):

    model = DLA(
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        block="BasicBlock",
        **kwargs
    )

    return model