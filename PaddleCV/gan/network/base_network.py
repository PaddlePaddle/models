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


def norm_layer(input,
               norm_type='batch_norm',
               name=None,
               is_test=False,
               affine=True):
    if norm_type == 'batch_norm':
        if affine == True:
            param_attr = fluid.ParamAttr(
                name=name + '_w',
                initializer=fluid.initializer.Normal(
                    loc=1.0, scale=0.02))
            bias_attr = fluid.ParamAttr(
                name=name + '_b',
                initializer=fluid.initializer.Constant(value=0.0))
        else:
            param_attr = fluid.ParamAttr(
                name=name + '_w',
                initializer=fluid.initializer.Constant(1.0),
                trainable=False)
            bias_attr = fluid.ParamAttr(
                name=name + '_b',
                initializer=fluid.initializer.Constant(value=0.0),
                trainable=False)
        return fluid.layers.batch_norm(
            input,
            param_attr=param_attr,
            bias_attr=bias_attr,
            is_test=is_test,
            moving_mean_name=name + '_mean',
            moving_variance_name=name + '_var')

    elif norm_type == 'instance_norm':
        if name is not None:
            scale_name = name + "_scale"
            offset_name = name + "_offset"
        if affine:
            scale_param = fluid.ParamAttr(
                name=scale_name,
                initializer=fluid.initializer.Constant(1.0),
                trainable=True)
            offset_param = fluid.ParamAttr(
                name=offset_name,
                initializer=fluid.initializer.Constant(0.0),
                trainable=True)
        else:
            scale_param = fluid.ParamAttr(
                name=scale_name,
                initializer=fluid.initializer.Constant(1.0),
                trainable=False)
            offset_param = fluid.ParamAttr(
                name=offset_name,
                initializer=fluid.initializer.Constant(0.0),
                trainable=False)
        return fluid.layers.instance_norm(
            input, param_attr=scale_param, bias_attr=offset_param)
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
           relufactor=0.2,
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
        left_padding, right_padding = cal_padding(input.shape[3], stride,
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
             relufactor=0.2,
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
        left_padding, right_padding = cal_padding(input.shape[3], stride,
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
    batch = fluid.layers.shape(x)[0]
    ones = fluid.layers.fill_constant(
        shape=[batch, y.shape[1], x.shape[2], x.shape[3]],
        dtype="float32",
        value=1.0)
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


def conv2d_spectral_norm(input,
                         num_filters=64,
                         filter_size=7,
                         stride=1,
                         stddev=0.02,
                         padding=0,
                         name="conv2d_spectral_norm",
                         norm=None,
                         activation_fn=None,
                         relufactor=0.0,
                         use_bias=False,
                         padding_type=None,
                         initial="normal",
                         is_test=False,
                         norm_affine=True):
    b, c, h, w = input.shape
    height = num_filters
    width = c * filter_size * filter_size
    helper = fluid.layer_helper.LayerHelper("conv2d_spectral_norm", **locals())
    dtype = helper.input_dtype()
    weight_param = fluid.ParamAttr(
        name=name + ".weight_orig",
        initializer=fluid.initializer.Normal(
            loc=0.0, scale=1.0),
        trainable=True)
    weight = helper.create_parameter(
        attr=weight_param,
        shape=(num_filters, c, filter_size, filter_size),
        dtype=dtype)
    weight_spectral_norm = fluid.layers.spectral_norm(
        weight, dim=0, name=name + ".spectral_norm")
    weight = weight_spectral_norm
    if use_bias:
        bias_attr = fluid.ParamAttr(
            name=name + "_b",
            initializer=fluid.initializer.Normal(
                loc=0.0, scale=1.0))
    else:
        bias_attr = False
    conv = conv2d_with_filter(
        input, weight, stride, padding, bias_attr=bias_attr, name=name)
    if norm is not None:
        conv = norm_layer(
            input=conv,
            norm_type=norm,
            name=name + "_norm",
            is_test=is_test,
            affine=norm_affine)
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


def conv2d_with_filter(input,
                       filter,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=None,
                       bias_attr=None,
                       use_cudnn=True,
                       act=None,
                       name=None):
    """ 
    Similar with conv2d, this is a convolution2D layers. Difference
    is filter can be token as input directly instead of setting filter size
    and number of fliters. Filter is a  4-D tensor with shape 
    [num_filter, num_channel, filter_size_h, filter_size_w].
     Args:
        input (Variable): The input image with [N, C, H, W] format.
        filter(Variable): The input filter with [N, C, H, W] format.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None
    Returns:
        Variable: The tensor variable storing the convolution and \
                  non-linearity activation result.
    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.
    Examples:
        .. code-block:: python
          data = fluid.data(name='data', shape=[None, 3, 32, 32], \
                                  dtype='float32')
          filter = fluid.data(name='filter',shape=[None, 10, 3, 3, 3], \
                                    dtype='float32',append_batch_size=False)
          conv2d = fluid.layers.conv2d(input=data, 
                                       filter=filter,
                                       act="relu") 
    """
    helper = fluid.layer_helper.LayerHelper("conv2d_with_filter", **locals())
    num_channels = input.shape[1]
    num_filters = filter.shape[0]
    num_filter_channels = filter.shape[1]
    l_type = 'conv2d'
    if (num_channels == groups and num_filters % num_channels == 0 and
            not use_cudnn):
        l_type = 'depthwise_conv2d'
    if groups is None:
        assert num_filter_channels == num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        if num_channels // groups != num_filter_channels:
            raise ValueError("num_filter_channels must equal to num_channels\
                              divided by groups.")
    stride = fluid.layers.utils.convert_to_list(stride, 2, 'stride')
    padding = fluid.layers.utils.convert_to_list(padding, 2, 'padding')
    dilation = fluid.layers.utils.convert_to_list(dilation, 2, 'dilation')
    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")
    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False
        })
    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    return helper.append_activation(pre_act)
