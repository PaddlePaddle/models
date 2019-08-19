#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import division
import paddle.fluid as fluid
import numpy as np
import math
import os
import warnings

use_cudnn = True
if 'ce_mode' in os.environ:
    use_cudnn = False


def cal_padding(img_size, stride, filter_size, dilation=1):
    """Calculate padding size."""
    valid_filter_size = dilation * (filter_size - 1) + 1
    if img_size % stride == 0:
        out_size = max(filter_size - stride, 0)
    else:
        out_size = max(filter_size - (img_size % stride), 0)
    return out_size // 2, out_size - out_size // 2


def norm_layer(input, norm_type='batch_norm', name=None, is_test=False):
    if norm_type == 'batch_norm':
        param_attr = fluid.ParamAttr(
            name=name + '_w', initializer=fluid.initializer.Constant(1.0))
        bias_attr = fluid.ParamAttr(
            name=name + '_b', initializer=fluid.initializer.Constant(value=0.0))
        return fluid.layers.batch_norm(
            input,
            param_attr=param_attr,
            bias_attr=bias_attr,
            is_test=is_test,
            moving_mean_name=name + '_mean',
            moving_variance_name=name + '_var')

    elif norm_type == 'instance_norm':
        helper = fluid.layer_helper.LayerHelper("instance_norm", **locals())
        dtype = helper.input_dtype()
        epsilon = 1e-5
        mean = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
        var = fluid.layers.reduce_mean(
            fluid.layers.square(input - mean), dim=[2, 3], keep_dim=True)
        if name is not None:
            scale_name = name + "_scale"
            offset_name = name + "_offset"
        scale_param = fluid.ParamAttr(
            name=scale_name,
            initializer=fluid.initializer.Constant(1.0),
            trainable=True)
        offset_param = fluid.ParamAttr(
            name=offset_name,
            initializer=fluid.initializer.Constant(0.0),
            trainable=True)
        scale = helper.create_parameter(
            attr=scale_param, shape=input.shape[1:2], dtype=dtype)
        offset = helper.create_parameter(
            attr=offset_param, shape=input.shape[1:2], dtype=dtype)

        tmp = fluid.layers.elementwise_mul(x=(input - mean), y=scale, axis=1)
        tmp = tmp / fluid.layers.sqrt(var + epsilon)
        tmp = fluid.layers.elementwise_add(tmp, offset, axis=1)
        return tmp
    else:
        raise NotImplementedError("norm type: [%s] is not support" % norm_type)


def initial_type(name,
                 input,
                 op_type,
                 fan_out,
                 init="normal",
                 use_bias=False,
                 filter_size=0,
                 stddev=0.02):
    if init == "kaiming":
        if op_type == 'conv':
            fan_in = input.shape[1] * filter_size * filter_size
        elif op_type == 'deconv':
            fan_in = fan_out * filter_size * filter_size
        else:
            if len(input.shape) > 2:
                fan_in = input.shape[1] * input.shape[2] * input.shape[3]
            else:
                fan_in = input.shape[1]
        bound = 1 / math.sqrt(fan_in)
        param_attr = fluid.ParamAttr(
            name=name + "_w",
            initializer=fluid.initializer.Uniform(
                low=-bound, high=bound))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + '_b',
                initializer=fluid.initializer.Uniform(
                    low=-bound, high=bound))
        else:
            bias_attr = False
    else:
        param_attr = fluid.ParamAttr(
            name=name + "_w",
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=stddev))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + "_b", initializer=fluid.initializer.Constant(0.0))
        else:
            bias_attr = False
    return param_attr, bias_attr


def conv2d(input,
           num_filters=64,
           filter_size=7,
           stride=1,
           stddev=0.02,
           padding=0,
           name="conv2d",
           norm=None,
           activation_fn=None,
           relufactor=0.0,
           use_bias=False,
           padding_type=None,
           initial="normal",
           is_test=False):

    if padding != 0 and padding_type != None:
        warnings.warn(
            'padding value and padding type are set in the same time, and the final padding width and padding height are computed by padding_type'
        )

    param_attr, bias_attr = initial_type(
        name=name,
        input=input,
        op_type='conv',
        fan_out=num_filters,
        init=initial,
        use_bias=use_bias,
        filter_size=filter_size,
        stddev=stddev)

    need_crop = False
    if padding_type == "SAME":
        top_padding, bottom_padding = cal_padding(input.shape[2], stride,
                                                  filter_size)
        left_padding, right_padding = cal_padding(input.shape[2], stride,
                                                  filter_size)
        height_padding = bottom_padding
        width_padding = right_padding
        if top_padding != bottom_padding or left_padding != right_padding:
            height_padding = top_padding + stride
            width_padding = left_padding + stride
            need_crop = True
        padding = [height_padding, width_padding]
    elif padding_type == "VALID":
        height_padding = 0
        width_padding = 0
        padding = [height_padding, width_padding]
    else:
        padding = padding

    conv = fluid.layers.conv2d(
        input,
        num_filters,
        filter_size,
        name=name,
        stride=stride,
        padding=padding,
        use_cudnn=use_cudnn,
        param_attr=param_attr,
        bias_attr=bias_attr)
    if need_crop:
        conv = fluid.layers.crop(
            conv,
            shape=(-1, conv.shape[1], conv.shape[2] - 1, conv.shape[3] - 1),
            offsets=(0, 0, 1, 1))
    if norm is not None:
        conv = norm_layer(
            input=conv, norm_type=norm, name=name + "_norm", is_test=is_test)
    if activation_fn == 'relu':
        conv = fluid.layers.relu(conv, name=name + '_relu')
    elif activation_fn == 'leaky_relu':
        conv = fluid.layers.leaky_relu(
            conv, alpha=relufactor, name=name + '_leaky_relu')
    elif activation_fn == 'tanh':
        conv = fluid.layers.tanh(conv, name=name + '_tanh')
    elif activation_fn == 'sigmoid':
        conv = fluid.layers.sigmoid(conv, name=name + '_sigmoid')
    elif activation_fn == None:
        conv = conv
    else:
        raise NotImplementedError("activation: [%s] is not support" %
                                  activation_fn)

    return conv


def deconv2d(input,
             num_filters=64,
             filter_size=7,
             stride=1,
             stddev=0.02,
             padding=0,
             outpadding=[0, 0, 0, 0],
             name="deconv2d",
             norm=None,
             activation_fn=None,
             relufactor=0.0,
             use_bias=False,
             padding_type=None,
             output_size=None,
             initial="normal",
             is_test=False):

    if padding != 0 and padding_type != None:
        warnings.warn(
            'padding value and padding type are set in the same time, and the final padding width and padding height are computed by padding_type'
        )

    param_attr, bias_attr = initial_type(
        name=name,
        input=input,
        op_type='deconv',
        fan_out=num_filters,
        init=initial,
        use_bias=use_bias,
        filter_size=filter_size,
        stddev=stddev)

    need_crop = False
    if padding_type == "SAME":
        top_padding, bottom_padding = cal_padding(input.shape[2], stride,
                                                  filter_size)
        left_padding, right_padding = cal_padding(input.shape[2], stride,
                                                  filter_size)
        height_padding = bottom_padding
        width_padding = right_padding
        if top_padding != bottom_padding or left_padding != right_padding:
            height_padding = top_padding + stride
            width_padding = left_padding + stride
            need_crop = True
        padding = [height_padding, width_padding]
    elif padding_type == "VALID":
        height_padding = 0
        width_padding = 0
        padding = [height_padding, width_padding]
    else:
        padding = padding

    conv = fluid.layers.conv2d_transpose(
        input,
        num_filters,
        output_size=output_size,
        name=name,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        use_cudnn=use_cudnn,
        param_attr=param_attr,
        bias_attr=bias_attr)

    if np.mean(outpadding) != 0 and padding_type == None:
        conv = fluid.layers.pad2d(
            conv, paddings=outpadding, mode='constant', pad_value=0.0)

    if norm is not None:
        conv = norm_layer(
            input=conv, norm_type=norm, name=name + "_norm", is_test=is_test)
    if activation_fn == 'relu':
        conv = fluid.layers.relu(conv, name=name + '_relu')
    elif activation_fn == 'leaky_relu':
        if relufactor == 0.0:
            raise Warning(
                "the activation is leaky_relu, but the relufactor is 0")
        conv = fluid.layers.leaky_relu(
            conv, alpha=relufactor, name=name + '_leaky_relu')
    elif activation_fn == 'tanh':
        conv = fluid.layers.tanh(conv, name=name + '_tanh')
    elif activation_fn == 'sigmoid':
        conv = fluid.layers.sigmoid(conv, name=name + '_sigmoid')
    elif activation_fn == None:
        conv = conv
    else:
        raise NotImplementedError("activation: [%s] is not support" %
                                  activation_fn)

    return conv


def linear(input,
           output_size,
           norm=None,
           stddev=0.02,
           activation_fn=None,
           relufactor=0.2,
           name="linear",
           initial="normal",
           is_test=False):

    param_attr, bias_attr = initial_type(
        name=name,
        input=input,
        op_type='linear',
        fan_out=output_size,
        init=initial,
        use_bias=True,
        filter_size=1,
        stddev=stddev)

    linear = fluid.layers.fc(input,
                             output_size,
                             param_attr=param_attr,
                             bias_attr=bias_attr,
                             name=name)

    if norm is not None:
        linear = norm_layer(
            input=linear, norm_type=norm, name=name + '_norm', is_test=is_test)
    if activation_fn == 'relu':
        linear = fluid.layers.relu(linear, name=name + '_relu')
    elif activation_fn == 'leaky_relu':
        if relufactor == 0.0:
            raise Warning(
                "the activation is leaky_relu, but the relufactor is 0")
        linear = fluid.layers.leaky_relu(
            linear, alpha=relufactor, name=name + '_leaky_relu')
    elif activation_fn == 'tanh':
        linear = fluid.layers.tanh(linear, name=name + '_tanh')
    elif activation_fn == 'sigmoid':
        linear = fluid.layers.sigmoid(linear, name=name + '_sigmoid')
    elif activation_fn == None:
        linear = linear
    else:
        raise NotImplementedError("activation: [%s] is not support" %
                                  activation_fn)

    return linear


def conv_cond_concat(x, y):
    ones = fluid.layers.fill_constant_batch_size_like(
        x, [-1, y.shape[1], x.shape[2], x.shape[3]], "float32", 1.0)
    out = fluid.layers.concat([x, ones * y], 1)
    return out


def conv_and_pool(x, num_filters, name, stddev=0.02, act=None):
    param_attr = fluid.ParamAttr(
        name=name + '_w',
        initializer=fluid.initializer.NormalInitializer(
            loc=0.0, scale=stddev))
    bias_attr = fluid.ParamAttr(
        name=name + "_b", initializer=fluid.initializer.Constant(0.0))

    out = fluid.nets.simple_img_conv_pool(
        input=x,
        filter_size=5,
        num_filters=num_filters,
        pool_size=2,
        pool_stride=2,
        param_attr=param_attr,
        bias_attr=bias_attr,
        act=act)
    return out
