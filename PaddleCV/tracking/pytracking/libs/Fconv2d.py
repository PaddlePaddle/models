from __future__ import print_function

import numpy as np
from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core, dygraph_utils
from paddle.fluid.layers import nn, utils
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper


def _is_list_or_tuple(input):
    return isinstance(input, (list, tuple))


def _zero_padding_in_batch_and_channel(padding, channel_last):
    if channel_last:
        return list(padding[0]) == [0, 0] and list(padding[-1]) == [0, 0]
    else:
        return list(padding[0]) == [0, 0] and list(padding[1]) == [0, 0]


def _exclude_padding_in_batch_and_channel(padding, channel_last):
    padding_ = padding[1:-1] if channel_last else padding[2:]
    padding_ = [elem for pad_a_dim in padding_ for elem in pad_a_dim]
    return padding_


def _update_padding_nd(padding, channel_last, num_dims):
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '{}'. It can only be 'SAME' or 'VALID'.".
                format(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0] * num_dims
        else:
            padding_algorithm = "SAME"
            padding = [0] * num_dims
    elif _is_list_or_tuple(padding):
        # for padding like
        # [(pad_before, pad_after), (pad_before, pad_after), ...]
        # padding for batch_dim and channel_dim included
        if len(padding) == 2 + num_dims and _is_list_or_tuple(padding[0]):
            if not _zero_padding_in_batch_and_channel(padding, channel_last):
                raise ValueError(
                    "Non-zero padding({}) in the batch or channel dimensions "
                    "is not supported.".format(padding))
            padding_algorithm = "EXPLICIT"
            padding = _exclude_padding_in_batch_and_channel(padding,
                                                            channel_last)
            if utils._is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_before, pad_after, pad_before, pad_after, ...]
        elif len(padding) == 2 * num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = utils.convert_to_list(padding, 2 * num_dims, 'padding')
            if utils._is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_d1, pad_d2, ...]
        elif len(padding) == num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = utils.convert_to_list(padding, num_dims, 'padding')
        else:
            raise ValueError("In valid padding: {}".format(padding))
    # for integer padding
    else:
        padding_algorithm = "EXPLICIT"
        padding = utils.convert_to_list(padding, num_dims, 'padding')
    return padding, padding_algorithm


def FConv2D(input,
           weight,
           bias=None,
           padding=0,
           stride=1,
           dilation=1,
           groups=1,
           use_cudnn=True,
           act=None,
           data_format="NCHW",
           name=None):
    # entry checks
    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. "
                         "Received Attr(use_cudnn): {}.".format(use_cudnn))
    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError("Attr(data_format) should be 'NCHW' or 'NHWC'. "
                         "Received Attr(data_format): {}.".format(data_format))

    channel_last = (data_format == "NHWC")
    channel_dim = -1 if channel_last else 1
    num_channels = input.shape[channel_dim]
    num_filters = weight.shape[0]
    if num_channels < 0:
        raise ValueError("The channel dimmention of the input({}) "
                         "should be defined. Received: {}.".format(
                             input.shape, num_channels))
    if num_channels % groups != 0:
        raise ValueError(
            "the channel of input must be divisible by groups,"
            "received: the channel of input is {}, the shape of input is {}"
            ", the groups is {}".format(num_channels, input.shape, groups))
    if num_filters % groups != 0:
        raise ValueError(
            "the number of filters must be divisible by groups,"
            "received: the number of filters is {}, the shape of weight is {}"
            ", the groups is {}".format(num_filters, weight.shape, groups))

    # update attrs
    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 2)
    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    l_type = "conv2d"
    if (num_channels == groups and num_filters % num_channels == 0 and
            not use_cudnn):
        l_type = 'depthwise_conv2d'

    inputs = {'Input': [input], 'Filter': [weight]}
    attrs = {
        'strides': stride,
        'paddings': padding,
        'dilations': dilation,
        'groups': groups,
        'use_cudnn': use_cudnn,
        'use_mkldnn': False,
        'fuse_relu_before_depthwise_conv': False,
        "padding_algorithm": padding_algorithm,
        "data_format": data_format
    }

    if in_dygraph_mode():
        attrs = ('strides', stride, 'paddings', padding, 'dilations', dilation,
                 'groups', groups, 'use_cudnn', use_cudnn, 'use_mkldnn', False,
                 'fuse_relu_before_depthwise_conv', False, "padding_algorithm",
                 padding_algorithm, "data_format", data_format)
        pre_bias = getattr(core.ops, l_type)(input, weight, *attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = dygraph_utils._append_activation_in_dygraph(
            pre_act, act, use_cudnn=use_cudnn)
    else:
        inputs = {'Input': [input], 'Filter': [weight]}
        attrs = {
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False,
            'fuse_relu_before_depthwise_conv': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format
        }
        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'], 'conv2d')
        helper = LayerHelper(l_type, **locals())
        dtype = helper.input_dtype()
        pre_bias = helper.create_variable_for_type_inference(dtype)
        outputs = {"Output": [pre_bias]}
        helper.append_op(
            type=l_type, inputs=inputs, outputs=outputs, attrs=attrs)
        if bias is not None:
            pre_act = nn.elementwise_add(pre_bias, bias, axis=channel_dim)
        else:
            pre_act = pre_bias
        out = helper.append_activation(pre_act)
    return out


def  test_conv2d_with_filter():

    import paddle.fluid.dygraph as dygraph
    import numpy as np

    exemplar = np.random.random((8, 4, 6, 6)).astype(np.float32)
    instance = np.random.random((8, 4, 22, 22)).astype(np.float32)

    with dygraph.guard():
        exem = dygraph.to_variable(exemplar)
        inst = dygraph.to_variable(instance)
        res = FConv2D(inst, exem, groups=1)
        print(res.shape)