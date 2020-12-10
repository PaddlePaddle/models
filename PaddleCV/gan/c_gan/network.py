#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
from utility import get_parent_function_name
import os

gf_dim = 64
df_dim = 64
gfc_dim = 1024 * 2
dfc_dim = 1024
img_dim = 28

c_dim = 3
y_dim = 1
output_height = 28
output_width = 28

use_cudnn = True
if 'ce_mode' in os.environ:
    use_cudnn = False


def bn(x, name=None, act='relu'):
    if name is None:
        name = get_parent_function_name()
    #return fluid.layers.leaky_relu(x)
    return fluid.layers.batch_norm(
        x,
        param_attr=name + '1',
        bias_attr=name + '2',
        moving_mean_name=name + '3',
        moving_variance_name=name + '4',
        name=name,
        act=act)


def conv(x, num_filters, name=None, act=None):
    if name is None:
        name = get_parent_function_name()
    return fluid.nets.simple_img_conv_pool(
        input=x,
        filter_size=5,
        num_filters=num_filters,
        pool_size=2,
        pool_stride=2,
        param_attr=name + 'w',
        bias_attr=name + 'b',
        use_cudnn=use_cudnn,
        act=act)


def fc(x, num_filters, name=None, act=None):
    if name is None:
        name = get_parent_function_name()
    return fluid.layers.fc(input=x,
                           size=num_filters,
                           act=act,
                           param_attr=name + 'w',
                           bias_attr=name + 'b')


def deconv(x,
           num_filters,
           name=None,
           filter_size=5,
           stride=2,
           dilation=1,
           padding=2,
           output_size=None,
           act=None):
    if name is None:
        name = get_parent_function_name()
    return fluid.layers.conv2d_transpose(
        input=x,
        param_attr=name + 'w',
        bias_attr=name + 'b',
        num_filters=num_filters,
        output_size=output_size,
        filter_size=filter_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        use_cudnn=use_cudnn,
        act=act)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shape = fluid.layers.shape(x)
    ones = fluid.layers.fill_constant(
        shape=[x_shape[0], y.shape[1], x.shape[2], x.shape[3]],
        dtype='float32',
        value=1.0)
    return fluid.layers.concat([x, ones * y], 1)


def D_cond(image, y):
    image = fluid.layers.reshape(x=image, shape=[-1, 1, 28, 28])
    yb = fluid.layers.reshape(y, [-1, y_dim, 1, 1])
    x = conv_cond_concat(image, yb)

    h0 = conv(x, c_dim + y_dim, act="leaky_relu")
    h0 = conv_cond_concat(h0, yb)
    h1 = bn(conv(h0, df_dim + y_dim), act="leaky_relu")
    h1 = fluid.layers.flatten(h1, axis=1)

    h1 = fluid.layers.concat([h1, y], 1)

    h2 = bn(fc(h1, dfc_dim), act='leaky_relu')
    h2 = fluid.layers.concat([h2, y], 1)

    h3 = fc(h2, 1, act='sigmoid')
    return h3


def G_cond(z, y):
    s_h, s_w = output_height, output_width
    s_h2, s_h4 = int(s_h // 2), int(s_h // 4)
    s_w2, s_w4 = int(s_w // 2), int(s_w // 4)

    yb = fluid.layers.reshape(y, [-1, y_dim, 1, 1])  #NCHW

    z = fluid.layers.concat([z, y], 1)
    h0 = bn(fc(z, gfc_dim // 2), act='relu')
    h0 = fluid.layers.concat([h0, y], 1)

    h1 = bn(fc(h0, gf_dim * 2 * s_h4 * s_w4), act='relu')
    h1 = fluid.layers.reshape(h1, [-1, gf_dim * 2, s_h4, s_w4])

    h1 = conv_cond_concat(h1, yb)
    h2 = bn(deconv(h1, gf_dim * 2, output_size=[s_h2, s_w2]), act='relu')
    h2 = conv_cond_concat(h2, yb)
    h3 = deconv(h2, 1, output_size=[s_h, s_w], act='tanh')
    return fluid.layers.reshape(h3, shape=[-1, s_h * s_w])


def D(x):
    x = fluid.layers.reshape(x=x, shape=[-1, 1, 28, 28])
    x = conv(x, df_dim, act='leaky_relu')
    x = bn(conv(x, df_dim * 2), act='leaky_relu')
    x = bn(fc(x, dfc_dim), act='leaky_relu')
    x = fc(x, 1, act='sigmoid')
    return x


def G(x):
    x = bn(fc(x, gfc_dim))
    x = bn(fc(x, gf_dim * 2 * img_dim // 4 * img_dim // 4))
    x = fluid.layers.reshape(x, [-1, gf_dim * 2, img_dim // 4, img_dim // 4])
    x = deconv(x, gf_dim * 2, act='relu', output_size=[14, 14])
    x = deconv(x, 1, filter_size=5, padding=2, act='tanh', output_size=[28, 28])
    x = fluid.layers.reshape(x, shape=[-1, 28 * 28])
    return x
