from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
import math
import warnings

def initial_type(name,
                 input,
                 op_type,
                 fan_out,
                 init="google",
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
            name=name + "_weights",
            initializer=fluid.initializer.Uniform(
                low=-bound, high=bound))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + '_offset',
                initializer=fluid.initializer.Uniform(
                    low=-bound, high=bound))
        else:
            bias_attr = False
    elif init == 'google':
        n = filter_size * filter_size * fan_out
        param_attr = fluid.ParamAttr(
            name=name + "_weights",
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=math.sqrt(2.0 / n)))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + "_offset", initializer=fluid.initializer.Constant(0.0))
        else:
            bias_attr = False

    else:
        param_attr = fluid.ParamAttr(
            name=name + "_weights",
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=stddev))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + "_offset", initializer=fluid.initializer.Constant(0.0))
        else:
            bias_attr = False
    return param_attr, bias_attr

def cal_padding(img_size, stride, filter_size, dilation=1):
    """Calculate padding size."""
    if img_size % stride == 0:
        out_size = max(filter_size - stride, 0)
    else:
        out_size = max(filter_size - (img_size % stride), 0)
    return out_size // 2, out_size - out_size // 2

def init_batch_norm_layer(name="batch_norm"):
    param_attr = fluid.ParamAttr(
        name=name + '_scale', initializer=fluid.initializer.Constant(1.0))
    bias_attr = fluid.ParamAttr(
        name=name + '_offset', initializer=fluid.initializer.Constant(value=0.0))
    return param_attr, bias_attr

def init_fc_layer(fout, name='fc'):
    n = fout  # fan-out
    init_range = 1.0 / math.sqrt(n)

    param_attr = fluid.ParamAttr(
        name=name + '_weights', initializer=fluid.initializer.UniformInitializer(
                low=-init_range, high=init_range))
    bias_attr = fluid.ParamAttr(
        name=name + '_offset', initializer=fluid.initializer.Constant(value=0.0))
    return param_attr, bias_attr

def norm_layer(input, norm_type='batch_norm', name=None):
    if norm_type == 'batch_norm':
        param_attr = fluid.ParamAttr(
            name=name + '_weights', initializer=fluid.initializer.Constant(1.0))
        bias_attr = fluid.ParamAttr(
            name=name + '_offset', initializer=fluid.initializer.Constant(value=0.0))
        return fluid.layers.batch_norm(
            input,
            param_attr=param_attr,
            bias_attr=bias_attr,
            moving_mean_name=name + '_mean',
            moving_variance_name=name + '_variance')

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
        raise NotImplementedError("norm tyoe: [%s] is not support" % norm_type)


def conv2d(input,
           num_filters=64,
           filter_size=7,
           stride=1,
           stddev=0.02,
           padding=0,
           groups=None,
           name="conv2d",
           norm=None,
           act=None,
           relufactor=0.0,
           use_bias=False,
           padding_type=None,
           initial="normal",
           use_cudnn=True):

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

    def get_padding(filter_size, stride=1, dilation=1):
        padding = ((stride - 1) + dilation * (filter_size - 1)) // 2
        return padding

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
    elif padding_type == "DYNAMIC":
        padding = get_padding(filter_size, stride)
    else:
        padding = padding

    conv = fluid.layers.conv2d(
        input,
        num_filters,
        filter_size,
        groups=groups,
        name=name,
        stride=stride,
        padding=padding,
        use_cudnn=use_cudnn,
        param_attr=param_attr,
        bias_attr=bias_attr)

    if need_crop:
        conv = conv[:, :, 1:, 1:]

    if norm is not None:
        conv = norm_layer(input=conv, norm_type=norm, name=name + "_norm")
    if act == 'relu':
        conv = fluid.layers.relu(conv, name=name + '_relu')
    elif act == 'leaky_relu':
        conv = fluid.layers.leaky_relu(
            conv, alpha=relufactor, name=name + '_leaky_relu')
    elif act == 'tanh':
        conv = fluid.layers.tanh(conv, name=name + '_tanh')
    elif act == 'sigmoid':
        conv = fluid.layers.sigmoid(conv, name=name + '_sigmoid')
    elif act == 'swish':
        conv = fluid.layers.swish(conv, name=name + '_swish')
    elif act == None:
        conv = conv
    else:
        raise NotImplementedError("activation: [%s] is not support" %act)

    return conv