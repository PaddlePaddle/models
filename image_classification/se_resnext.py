import paddle.v2 as paddle

__all__ = ['se_resnext50']


def squeeze_excitation(input,
                       num_channels,
                       pool_size,
                       reduction_ratio=16,
                       name='__SE'):
    squeeze = paddle.layer.img_pool(
        name='{0}_globalpool'.format(name),
        input=input,
        pool_size=pool_size,
        stride=1,
        num_channels=num_channels,
        pool_type=paddle.pooling.Avg())
    squeeze = paddle.layer.fc(
        name='{0}_fc0'.format(name),
        input=squeeze,
        size=num_channels / reduction_ratio,
        act=paddle.activation.Relu())
    excitation = paddle.layer.fc(
        name='{0}_fc1'.format(name),
        input=squeeze,
        size=num_channels,
        act=paddle.activation.Sigmoid())
    scale = paddle.layer.broadcast_scale(input=input, weight=excitation)
    return scale


def se_resnext50(input, class_dim):
    conv0 = paddle.layer.img_conv(
        name='conv0',
        input=input,
        num_channels=3,
        num_filters=64,
        filter_size=7,
        padding=(7 - 1) / 2,
        stride=2,
        act=paddle.activation.Linear())
    conv0 = paddle.layer.batch_norm(
        name='conv0_norm', input=conv0, act=paddle.activation.Relu())
    pool0 = paddle.layer.img_pool(
        name='resnext_pool0',
        input=conv0,
        pool_size=3,
        stride=2,
        num_channels=64,
        pool_type=paddle.pooling.Max())

    def conv_block(input, group, depth, input_channels, num_filters, stride,
                   cardinality, out_size):
        conv0 = paddle.layer.img_conv(
            name='conv{0}_{1}_0'.format(group, depth),
            input=input,
            num_channels=input_channels,
            num_filters=num_filters,
            filter_size=1,
            act=paddle.activation.Linear())
        conv0 = paddle.layer.batch_norm(
            name='conv{0}_{1}_0_norm'.format(group, depth),
            input=conv0,
            act=paddle.activation.Relu())
        conv1 = paddle.layer.img_conv(
            name='conv{0}_{1}_1'.format(group, depth),
            input=conv0,
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            padding=1,
            stride=stride,
            groups=cardinality,
            act=paddle.activation.Linear())
        conv1 = paddle.layer.batch_norm(
            name='conv{0}_{1}_1_norm'.format(group, depth),
            input=conv1,
            act=paddle.activation.Relu())
        conv2 = paddle.layer.img_conv(
            name='conv{0}_{1}_2'.format(group, depth),
            input=conv1,
            num_channels=num_filters,
            num_filters=num_filters * 2,
            filter_size=1,
            act=paddle.activation.Linear())
        conv2 = paddle.layer.batch_norm(
            name='conv{0}_{1}_2_norm'.format(group, depth),
            input=conv2,
            act=paddle.activation.Linear())

        scale = squeeze_excitation(
            name='SE{0}_{1}'.format(group, depth),
            input=conv2,
            num_channels=num_filters * 2,
            pool_size=out_size)

        if input_channels == num_filters * 2:
            shortcut = input
        else:
            shortcut = paddle.layer.img_conv(
                name='shortcut_proj_{0}'.format(group),
                input=input,
                num_channels=input_channels,
                num_filters=num_filters * 2,
                filter_size=1,
                stride=stride,
                act=paddle.activation.Linear())
            shortcut = paddle.layer.batch_norm(
                name='shortcut_proj_{0}_norm'.format(group),
                input=shortcut,
                act=paddle.activation.Linear())

        return paddle.layer.addto(
            input=[scale, shortcut], act=paddle.activation.Relu())

    depth = [3, 4, 6, 3]
    num_filters = [128, 256, 512, 1024]
    input_channels = [64, 256, 512, 1024]
    strides = [1, 2, 2, 2]
    out_size = [56, 28, 14, 7]
    conv = pool0
    for group in range(4):
        for i in range(depth[group]):
            conv = conv_block(
                input=conv,
                group=group + 1,
                depth=i,
                input_channels=input_channels[group]
                if i == 0 else num_filters[group] * 2,
                num_filters=num_filters[group],
                stride=strides[group] if i == 0 else 1,
                cardinality=32,
                out_size=out_size[group])

    pool1 = paddle.layer.img_pool(
        name='resnext_globalpool',
        input=conv,
        pool_size=7,
        stride=1,
        num_channels=2048,
        pool_type=paddle.pooling.Avg())

    out = paddle.layer.fc(
        name='resnext_fc',
        input=pool1,
        size=class_dim,
        act=paddle.activation.Softmax())
    return out
