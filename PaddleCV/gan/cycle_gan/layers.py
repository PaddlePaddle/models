from __future__ import division
import paddle.fluid as fluid
import numpy as np
import os

# cudnn is not better when batch size is 1.
use_cudnn = False
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


def instance_norm(input, name=None):
    # TODO(lvmengsi@baidu.com): Check the accuracy when using fluid.layers.layer_norm.
    # return fluid.layers.layer_norm(input, begin_norm_axis=2) 
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
        initializer=fluid.initializer.TruncatedNormal(1.0, 0.02),
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


def conv2d(input,
           num_filters=64,
           filter_size=7,
           stride=1,
           stddev=0.02,
           padding="VALID",
           name="conv2d",
           norm=True,
           relu=True,
           relufactor=0.0):
    """Wrapper for conv2d op to support VALID and SAME padding mode."""
    need_crop = False
    if padding == "SAME":
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
    else:
        height_padding = 0
        width_padding = 0

    padding = [height_padding, width_padding]
    param_attr = fluid.ParamAttr(
        name=name + "_w",
        initializer=fluid.initializer.TruncatedNormal(scale=stddev))
    bias_attr = fluid.ParamAttr(
        name=name + "_b", initializer=fluid.initializer.Constant(0.0))
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
    if norm:
        conv = instance_norm(input=conv, name=name + "_norm")
    if relu:
        conv = fluid.layers.leaky_relu(conv, alpha=relufactor)
    return conv


def deconv2d(input,
             out_shape,
             num_filters=64,
             filter_size=7,
             stride=1,
             stddev=0.02,
             padding="VALID",
             name="conv2d",
             norm=True,
             relu=True,
             relufactor=0.0):
    """Wrapper for deconv2d op to support VALID and SAME padding mode."""
    need_crop = False
    if padding == "SAME":
        top_padding, bottom_padding = cal_padding(out_shape[0], stride,
                                                  filter_size)
        left_padding, right_padding = cal_padding(out_shape[1], stride,
                                                  filter_size)
        height_padding = top_padding
        width_padding = left_padding
        if top_padding != bottom_padding or left_padding != right_padding:
            need_crop = True
    else:
        height_padding = 0
        width_padding = 0

    padding = [height_padding, width_padding]

    param_attr = fluid.ParamAttr(
        name=name + "_w",
        initializer=fluid.initializer.TruncatedNormal(scale=stddev))
    bias_attr = fluid.ParamAttr(
        name=name + "_b", initializer=fluid.initializer.Constant(0.0))
    conv = fluid.layers.conv2d_transpose(
        input,
        num_filters,
        name=name,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        use_cudnn=use_cudnn,
        param_attr=param_attr,
        bias_attr=bias_attr)

    if need_crop:
        conv = fluid.layers.crop(
            conv,
            shape=(-1, conv.shape[1], conv.shape[2] - 1, conv.shape[3] - 1),
            offsets=(0, 0, 0, 0))
    if norm:
        conv = instance_norm(input=conv, name=name + "_norm")
    if relu:
        conv = fluid.layers.leaky_relu(conv, alpha=relufactor)
    return conv
