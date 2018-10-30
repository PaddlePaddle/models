from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid

import contextlib
name_scope = ""

decode_channel = 48
encode_channel = 256
label_number = 19

bn_momentum = 0.99
dropout_keep_prop = 0.9
is_train = True

op_results = {}

default_epsilon = 1e-3
default_norm_type = 'bn'
default_group_number = 32


@contextlib.contextmanager
def scope(name):
    global name_scope
    bk = name_scope
    name_scope = name_scope + name + '/'
    yield
    name_scope = bk


def check(data, number):
    if type(data) == int:
        return [data] * number
    assert len(data) == number
    return data


def clean():
    global op_results
    op_results = {}


def append_op_result(result, name):
    global op_results
    op_index = len(op_results)
    name = name_scope + name + str(op_index)
    op_results[name] = result
    return result


def conv(*args, **kargs):
    kargs['param_attr'] = name_scope + 'weights'
    if 'bias_attr' in kargs and kargs['bias_attr']:
        kargs['bias_attr'] = name_scope + 'biases'
    else:
        kargs['bias_attr'] = False
    return append_op_result(fluid.layers.conv2d(*args, **kargs), 'conv')


def group_norm(input, G, eps=1e-5, param_attr=None, bias_attr=None):
    helper = fluid.layer_helper.LayerHelper('group_norm', **locals())

    N, C, H, W = input.shape
    if C % G != 0:
        print("group can not divide channle:", C, G)
        for d in range(10):
            for t in [d, -d]:
                if G + t <= 0: continue
                if C % (G + t) == 0:
                    G = G + t
                    break
            if C % G == 0:
                print("use group size:", G)
                break
    assert C % G == 0
    param_shape = (G, )
    x = input
    x = fluid.layers.reshape(x, [N, G, C // G * H * W])
    mean = fluid.layers.reduce_mean(x, dim=2, keep_dim=True)
    x = x - mean
    var = fluid.layers.reduce_mean(fluid.layers.square(x), dim=2, keep_dim=True)
    x = x / fluid.layers.sqrt(var + eps)

    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype='float32',
        default_initializer=fluid.initializer.Constant(1.0))

    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=param_shape, dtype='float32', is_bias=True)
    x = fluid.layers.elementwise_add(
        fluid.layers.elementwise_mul(
            x, scale, axis=1), bias, axis=1)
    return fluid.layers.reshape(x, input.shape)


def bn(*args, **kargs):
    if default_norm_type == 'bn':
        with scope('BatchNorm'):
            return append_op_result(
                fluid.layers.batch_norm(
                    *args,
                    epsilon=default_epsilon,
                    momentum=bn_momentum,
                    param_attr=name_scope + 'gamma',
                    bias_attr=name_scope + 'beta',
                    moving_mean_name=name_scope + 'moving_mean',
                    moving_variance_name=name_scope + 'moving_variance',
                    **kargs),
                'bn')
    elif default_norm_type == 'gn':
        with scope('GroupNorm'):
            return append_op_result(
                group_norm(
                    args[0],
                    default_group_number,
                    eps=default_epsilon,
                    param_attr=name_scope + 'gamma',
                    bias_attr=name_scope + 'beta'),
                'gn')
    else:
        raise "Unsupport norm type:" + default_norm_type


def bn_relu(data):
    return append_op_result(fluid.layers.relu(bn(data)), 'relu')


def relu(data):
    return append_op_result(fluid.layers.relu(data), 'relu')


def seq_conv(input, channel, stride, filter, dilation=1, act=None):
    with scope('depthwise'):
        input = conv(
            input,
            input.shape[1],
            filter,
            stride,
            groups=input.shape[1],
            padding=(filter // 2) * dilation,
            dilation=dilation)
        input = bn(input)
        if act: input = act(input)
    with scope('pointwise'):
        input = conv(input, channel, 1, 1, groups=1, padding=0)
        input = bn(input)
        if act: input = act(input)
    return input


def xception_block(input,
                   channels,
                   strides=1,
                   filters=3,
                   dilation=1,
                   skip_conv=True,
                   has_skip=True,
                   activation_fn_in_separable_conv=False):
    repeat_number = 3
    channels = check(channels, repeat_number)
    filters = check(filters, repeat_number)
    strides = check(strides, repeat_number)
    data = input
    datum = []
    for i in range(repeat_number):
        with scope('separable_conv' + str(i + 1)):
            if not activation_fn_in_separable_conv:
                data = relu(data)
                data = seq_conv(
                    data,
                    channels[i],
                    strides[i],
                    filters[i],
                    dilation=dilation)
            else:
                data = seq_conv(
                    data,
                    channels[i],
                    strides[i],
                    filters[i],
                    dilation=dilation,
                    act=relu)
            datum.append(data)
    if not has_skip:
        return append_op_result(data, 'xception_block'), datum
    if skip_conv:
        with scope('shortcut'):
            skip = bn(
                conv(
                    input, channels[-1], 1, strides[-1], groups=1, padding=0))
    else:
        skip = input
    return append_op_result(data + skip, 'xception_block'), datum


def entry_flow(data):
    with scope("entry_flow"):
        with scope("conv1"):
            data = conv(data, 32, 3, stride=2, padding=1)
            data = bn_relu(data)
        with scope("conv2"):
            data = conv(data, 64, 3, stride=1, padding=1)
            data = bn_relu(data)
        with scope("block1"):
            data, _ = xception_block(data, 128, [1, 1, 2])
        with scope("block2"):
            data, datum = xception_block(data, 256, [1, 1, 2])
        with scope("block3"):
            data, _ = xception_block(data, 728, [1, 1, 2])
        return data, datum[1]


def middle_flow(data):
    with scope("middle_flow"):
        for i in range(16):
            with scope("block" + str(i + 1)):
                data, _ = xception_block(data, 728, [1, 1, 1], skip_conv=False)
    return data


def exit_flow(data):
    with scope("exit_flow"):
        with scope('block1'):
            data, _ = xception_block(data, [728, 1024, 1024], [1, 1, 1])
        with scope('block2'):
            data, _ = xception_block(
                data, [1536, 1536, 2048], [1, 1, 1],
                dilation=2,
                has_skip=False,
                activation_fn_in_separable_conv=True)
        return data


def dropout(x, keep_rate):
    if is_train:
        return fluid.layers.dropout(x, 1 - keep_rate) / keep_rate
    else:
        return x


def encoder(input):
    with scope('encoder'):
        channel = 256
        with scope("image_pool"):
            image_avg = fluid.layers.reduce_mean(input, [2, 3], keep_dim=True)
            append_op_result(image_avg, 'reduce_mean')
            image_avg = bn_relu(
                conv(
                    image_avg, channel, 1, 1, groups=1, padding=0))
            image_avg = fluid.layers.resize_bilinear(image_avg, input.shape[2:])

        with scope("aspp0"):
            aspp0 = bn_relu(conv(input, channel, 1, 1, groups=1, padding=0))
        with scope("aspp1"):
            aspp1 = seq_conv(input, channel, 1, 3, dilation=6, act=relu)
        with scope("aspp2"):
            aspp2 = seq_conv(input, channel, 1, 3, dilation=12, act=relu)
        with scope("aspp3"):
            aspp3 = seq_conv(input, channel, 1, 3, dilation=18, act=relu)
        with scope("concat"):
            data = append_op_result(
                fluid.layers.concat(
                    [image_avg, aspp0, aspp1, aspp2, aspp3], axis=1),
                'concat')
            data = bn_relu(conv(data, channel, 1, 1, groups=1, padding=0))
            data = dropout(data, dropout_keep_prop)
        return data


def decoder(encode_data, decode_shortcut):
    with scope('decoder'):
        with scope('concat'):
            decode_shortcut = bn_relu(
                conv(
                    decode_shortcut, decode_channel, 1, 1, groups=1, padding=0))
            encode_data = fluid.layers.resize_bilinear(
                encode_data, decode_shortcut.shape[2:])
            encode_data = fluid.layers.concat(
                [encode_data, decode_shortcut], axis=1)
            append_op_result(encode_data, 'concat')
        with scope("separable_conv1"):
            encode_data = seq_conv(
                encode_data, encode_channel, 1, 3, dilation=1, act=relu)
        with scope("separable_conv2"):
            encode_data = seq_conv(
                encode_data, encode_channel, 1, 3, dilation=1, act=relu)
        return encode_data


def deeplabv3p(img):
    global default_epsilon
    append_op_result(img, 'img')
    with scope('xception_65'):
        default_epsilon = 1e-3
        # Entry flow
        data, decode_shortcut = entry_flow(img)
        # Middle flow
        data = middle_flow(data)
        # Exit flow
        data = exit_flow(data)
    default_epsilon = 1e-5
    encode_data = encoder(data)
    encode_data = decoder(encode_data, decode_shortcut)
    with scope('logit'):
        logit = conv(
            encode_data, label_number, 1, stride=1, padding=0, bias_attr=True)
        logit = fluid.layers.resize_bilinear(logit, img.shape[2:])
    return logit
