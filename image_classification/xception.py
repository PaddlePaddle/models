import paddle.v2 as paddle

__all__ = ['xception']


def img_separable_conv_bn(name, input, num_channels, num_out_channels,
                          filter_size, stride, padding, act):
    conv = paddle.networks.img_separable_conv(
        name=name,
        input=input,
        num_channels=num_channels,
        num_out_channels=num_out_channels,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear())
    norm = paddle.layer.batch_norm(name=name + '_norm', input=conv, act=act)
    return norm


def img_conv_bn(name, input, num_channels, num_filters, filter_size, stride,
                padding, act):
    conv = paddle.layer.img_conv(
        name=name,
        input=input,
        num_channels=num_channels,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear())
    norm = paddle.layer.batch_norm(name=name + '_norm', input=conv, act=act)
    return norm


def conv_block0(input,
                group,
                num_channels,
                num_filters,
                num_filters2=None,
                filter_size=3,
                pool_padding=0,
                entry_relu=True):
    if num_filters2 is None:
        num_filters2 = num_filters

    if entry_relu:
        act_input = paddle.layer.mixed(
            input=paddle.layer.identity_projection(input=input),
            act=paddle.activation.Relu())
    else:
        act_input = input
    conv0 = img_separable_conv_bn(
        name='xception_block{0}_conv0'.format(group),
        input=act_input,
        num_channels=num_channels,
        num_out_channels=num_filters,
        filter_size=filter_size,
        stride=1,
        padding=(filter_size - 1) / 2,
        act=paddle.activation.Relu())
    conv1 = img_separable_conv_bn(
        name='xception_block{0}_conv1'.format(group),
        input=conv0,
        num_channels=num_filters,
        num_out_channels=num_filters2,
        filter_size=filter_size,
        stride=1,
        padding=(filter_size - 1) / 2,
        act=paddle.activation.Linear())
    pool0 = paddle.layer.img_pool(
        name='xception_block{0}_pool'.format(group),
        input=conv1,
        pool_size=3,
        stride=2,
        padding=pool_padding,
        num_channels=num_filters2,
        pool_type=paddle.pooling.CudnnMax())

    shortcut = img_conv_bn(
        name='xception_block{0}_shortcut'.format(group),
        input=input,
        num_channels=num_channels,
        num_filters=num_filters2,
        filter_size=1,
        stride=2,
        padding=0,
        act=paddle.activation.Linear())

    return paddle.layer.addto(
        input=[pool0, shortcut], act=paddle.activation.Linear())


def conv_block1(input, group, num_channels, num_filters, filter_size=3):
    act_input = paddle.layer.mixed(
        input=paddle.layer.identity_projection(input=input),
        act=paddle.activation.Relu())
    conv0 = img_separable_conv_bn(
        name='xception_block{0}_conv0'.format(group),
        input=act_input,
        num_channels=num_channels,
        num_out_channels=num_filters,
        filter_size=filter_size,
        stride=1,
        padding=(filter_size - 1) / 2,
        act=paddle.activation.Relu())
    conv1 = img_separable_conv_bn(
        name='xception_block{0}_conv1'.format(group),
        input=conv0,
        num_channels=num_filters,
        num_out_channels=num_filters,
        filter_size=filter_size,
        stride=1,
        padding=(filter_size - 1) / 2,
        act=paddle.activation.Relu())
    conv2 = img_separable_conv_bn(
        name='xception_block{0}_conv2'.format(group),
        input=conv1,
        num_channels=num_filters,
        num_out_channels=num_filters,
        filter_size=filter_size,
        stride=1,
        padding=(filter_size - 1) / 2,
        act=paddle.activation.Linear())

    shortcut = input
    return paddle.layer.addto(
        input=[conv2, shortcut], act=paddle.activation.Linear())


def xception(input, class_dim):
    conv = img_conv_bn(
        name='xception_conv0',
        input=input,
        num_channels=3,
        num_filters=32,
        filter_size=3,
        stride=2,
        padding=1,
        act=paddle.activation.Relu())
    conv = img_conv_bn(
        name='xception_conv1',
        input=conv,
        num_channels=32,
        num_filters=64,
        filter_size=3,
        stride=1,
        padding=1,
        act=paddle.activation.Relu())
    conv = conv_block0(
        input=conv, group=2, num_channels=64, num_filters=128, entry_relu=False)
    conv = conv_block0(input=conv, group=3, num_channels=128, num_filters=256)
    conv = conv_block0(input=conv, group=4, num_channels=256, num_filters=728)
    for group in range(5, 13):
        conv = conv_block1(
            input=conv, group=group, num_channels=728, num_filters=728)
    conv = conv_block0(
        input=conv,
        group=13,
        num_channels=728,
        num_filters=728,
        num_filters2=1024)
    conv = img_separable_conv_bn(
        name='xception_conv14',
        input=conv,
        num_channels=1024,
        num_out_channels=1536,
        filter_size=3,
        stride=1,
        padding=1,
        act=paddle.activation.Relu())
    conv = img_separable_conv_bn(
        name='xception_conv15',
        input=conv,
        num_channels=1536,
        num_out_channels=2048,
        filter_size=3,
        stride=1,
        padding=1,
        act=paddle.activation.Relu())
    pool = paddle.layer.img_pool(
        name='xception_global_pool',
        input=conv,
        pool_size=7,
        stride=1,
        num_channels=2048,
        pool_type=paddle.pooling.CudnnAvg())
    out = paddle.layer.fc(name='xception_fc',
                          input=pool,
                          size=class_dim,
                          act=paddle.activation.Softmax())
    return out
