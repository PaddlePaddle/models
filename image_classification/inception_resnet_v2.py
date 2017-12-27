import paddle.v2 as paddle


def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding=0,
                  active_type=paddle.activation.Relu(),
                  ch_in=None):
    """layer wrapper assembling convolution and batchnorm layer"""
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False)
    return paddle.layer.batch_norm(input=tmp, epsilon=0.001, act=active_type)


def sequential_block(input, *layers):
    """helper function for sequential layers"""
    for layer in layers:
        layer_func, layer_conf = layer
        input = layer_func(input, **layer_conf)
    return input


def mixed_5b_block(input):
    branch0 = conv_bn_layer(
        input, ch_in=192, ch_out=96, filter_size=1, stride=1)
    branch1 = sequential_block(input, (conv_bn_layer, {
        "ch_in": 192,
        "ch_out": 48,
        "filter_size": 1,
        "stride": 1
    }), (conv_bn_layer, {
        "ch_in": 48,
        "ch_out": 64,
        "filter_size": 5,
        "stride": 1,
        "padding": 2
    }))
    branch2 = sequential_block(input, (conv_bn_layer, {
        "ch_in": 192,
        "ch_out": 64,
        "filter_size": 1,
        "stride": 1
    }), (conv_bn_layer, {
        "ch_in": 64,
        "ch_out": 96,
        "filter_size": 3,
        "stride": 1,
        "padding": 1
    }), (conv_bn_layer, {
        "ch_in": 96,
        "ch_out": 96,
        "filter_size": 3,
        "stride": 1,
        "padding": 1
    }))
    branch3 = sequential_block(
        input,
        (paddle.layer.img_pool, {
            "pool_size": 3,
            "stride": 1,
            "padding": 1,
            "pool_type": paddle.pooling.Avg(),
            "exclude_mode": False
        }),
        (conv_bn_layer, {
            "ch_in": 192,
            "ch_out": 64,
            "filter_size": 1,
            "stride": 1
        }), )
    out = paddle.layer.concat(input=[branch0, branch1, branch2, branch3])
    return out


def block35(input, scale=1.0):
    branch0 = conv_bn_layer(
        input, ch_in=320, ch_out=32, filter_size=1, stride=1)
    branch1 = sequential_block(input, (conv_bn_layer, {
        "ch_in": 320,
        "ch_out": 32,
        "filter_size": 1,
        "stride": 1
    }), (conv_bn_layer, {
        "ch_in": 32,
        "ch_out": 32,
        "filter_size": 3,
        "stride": 1,
        "padding": 1
    }))
    branch2 = sequential_block(input, (conv_bn_layer, {
        "ch_in": 320,
        "ch_out": 32,
        "filter_size": 1,
        "stride": 1
    }), (conv_bn_layer, {
        "ch_in": 32,
        "ch_out": 48,
        "filter_size": 3,
        "stride": 1,
        "padding": 1
    }), (conv_bn_layer, {
        "ch_in": 48,
        "ch_out": 64,
        "filter_size": 3,
        "stride": 1,
        "padding": 1
    }))
    out = paddle.layer.concat(input=[branch0, branch1, branch2])
    out = paddle.layer.img_conv(
        input=out,
        filter_size=1,
        num_channels=128,
        num_filters=320,
        stride=1,
        padding=0,
        act=paddle.activation.Linear(),
        bias_attr=None)
    out = paddle.layer.slope_intercept(out, slope=scale, intercept=0.0)
    out = paddle.layer.addto(input=[input, out], act=paddle.activation.Relu())
    return out


def mixed_6a_block(input):
    branch0 = conv_bn_layer(
        input, ch_in=320, ch_out=384, filter_size=3, stride=2)
    branch1 = sequential_block(input, (conv_bn_layer, {
        "ch_in": 320,
        "ch_out": 256,
        "filter_size": 1,
        "stride": 1
    }), (conv_bn_layer, {
        "ch_in": 256,
        "ch_out": 256,
        "filter_size": 3,
        "stride": 1,
        "padding": 1
    }), (conv_bn_layer, {
        "ch_in": 256,
        "ch_out": 384,
        "filter_size": 3,
        "stride": 2
    }))
    branch2 = paddle.layer.img_pool(
        input,
        num_channels=320,
        pool_size=3,
        stride=2,
        pool_type=paddle.pooling.Max())
    out = paddle.layer.concat(input=[branch0, branch1, branch2])
    return out


def block17(input, scale=1.0):
    branch0 = conv_bn_layer(
        input, ch_in=1088, ch_out=192, filter_size=1, stride=1)
    branch1 = sequential_block(input, (conv_bn_layer, {
        "ch_in": 1088,
        "ch_out": 128,
        "filter_size": 1,
        "stride": 1
    }), (conv_bn_layer, {
        "ch_in": 128,
        "ch_out": 160,
        "filter_size": [7, 1],
        "stride": 1,
        "padding": [3, 0]
    }), (conv_bn_layer, {
        "ch_in": 160,
        "ch_out": 192,
        "filter_size": [1, 7],
        "stride": 1,
        "padding": [0, 3]
    }))
    out = paddle.layer.concat(input=[branch0, branch1])
    out = paddle.layer.img_conv(
        input=out,
        filter_size=1,
        num_channels=384,
        num_filters=1088,
        stride=1,
        padding=0,
        act=paddle.activation.Linear(),
        bias_attr=None)
    out = paddle.layer.slope_intercept(out, slope=scale, intercept=0.0)
    out = paddle.layer.addto(input=[input, out], act=paddle.activation.Relu())
    return out


def mixed_7a_block(input):
    branch0 = sequential_block(
        input,
        (conv_bn_layer, {
            "ch_in": 1088,
            "ch_out": 256,
            "filter_size": 1,
            "stride": 1
        }),
        (conv_bn_layer, {
            "ch_in": 256,
            "ch_out": 384,
            "filter_size": 3,
            "stride": 2
        }), )
    branch1 = sequential_block(
        input,
        (conv_bn_layer, {
            "ch_in": 1088,
            "ch_out": 256,
            "filter_size": 1,
            "stride": 1
        }),
        (conv_bn_layer, {
            "ch_in": 256,
            "ch_out": 288,
            "filter_size": 3,
            "stride": 2
        }), )
    branch2 = sequential_block(input, (conv_bn_layer, {
        "ch_in": 1088,
        "ch_out": 256,
        "filter_size": 1,
        "stride": 1
    }), (conv_bn_layer, {
        "ch_in": 256,
        "ch_out": 288,
        "filter_size": 3,
        "stride": 1,
        "padding": 1
    }), (conv_bn_layer, {
        "ch_in": 288,
        "ch_out": 320,
        "filter_size": 3,
        "stride": 2
    }))
    branch3 = paddle.layer.img_pool(
        input,
        num_channels=1088,
        pool_size=3,
        stride=2,
        pool_type=paddle.pooling.Max())
    out = paddle.layer.concat(input=[branch0, branch1, branch2, branch3])
    return out


def block8(input, scale=1.0, no_relu=False):
    branch0 = conv_bn_layer(
        input, ch_in=2080, ch_out=192, filter_size=1, stride=1)
    branch1 = sequential_block(input, (conv_bn_layer, {
        "ch_in": 2080,
        "ch_out": 192,
        "filter_size": 1,
        "stride": 1
    }), (conv_bn_layer, {
        "ch_in": 192,
        "ch_out": 224,
        "filter_size": [3, 1],
        "stride": 1,
        "padding": [1, 0]
    }), (conv_bn_layer, {
        "ch_in": 224,
        "ch_out": 256,
        "filter_size": [1, 3],
        "stride": 1,
        "padding": [0, 1]
    }))
    out = paddle.layer.concat(input=[branch0, branch1])
    out = paddle.layer.img_conv(
        input=out,
        filter_size=1,
        num_channels=448,
        num_filters=2080,
        stride=1,
        padding=0,
        act=paddle.activation.Linear(),
        bias_attr=None)
    out = paddle.layer.slope_intercept(out, slope=scale, intercept=0.0)
    out = paddle.layer.addto(
        input=[input, out],
        act=paddle.activation.Linear() if no_relu else paddle.activation.Relu())
    return out


def inception_resnet_v2(input,
                        class_dim,
                        dropout_rate=0.5,
                        data_dim=3 * 331 * 331):
    conv2d_1a = conv_bn_layer(
        input, ch_in=3, ch_out=32, filter_size=3, stride=2)
    conv2d_2a = conv_bn_layer(
        conv2d_1a, ch_in=32, ch_out=32, filter_size=3, stride=1)
    conv2d_2b = conv_bn_layer(
        conv2d_2a, ch_in=32, ch_out=64, filter_size=3, stride=1, padding=1)
    maxpool_3a = paddle.layer.img_pool(
        input=conv2d_2b, pool_size=3, stride=2, pool_type=paddle.pooling.Max())
    conv2d_3b = conv_bn_layer(
        maxpool_3a, ch_in=64, ch_out=80, filter_size=1, stride=1)
    conv2d_4a = conv_bn_layer(
        conv2d_3b, ch_in=80, ch_out=192, filter_size=3, stride=1)
    maxpool_5a = paddle.layer.img_pool(
        input=conv2d_4a, pool_size=3, stride=2, pool_type=paddle.pooling.Max())
    mixed_5b = mixed_5b_block(maxpool_5a)
    repeat = sequential_block(mixed_5b, *([(block35, {"scale": 0.17})] * 10))
    mixed_6a = mixed_6a_block(repeat)
    repeat1 = sequential_block(mixed_6a, *([(block17, {"scale": 0.10})] * 20))
    mixed_7a = mixed_7a_block(repeat1)
    repeat2 = sequential_block(mixed_7a, *([(block8, {"scale": 0.20})] * 9))
    block_8 = block8(repeat2, no_relu=True)
    conv2d_7b = conv_bn_layer(
        block_8, ch_in=2080, ch_out=1536, filter_size=1, stride=1)
    avgpool_1a = paddle.layer.img_pool(
        input=conv2d_7b,
        pool_size=8 if data_dim == 3 * 299 * 299 else 9,
        stride=1,
        pool_type=paddle.pooling.Avg(),
        exclude_mode=False)
    drop_out = paddle.layer.dropout(input=avgpool_1a, dropout_rate=dropout_rate)
    out = paddle.layer.fc(
        input=drop_out, size=class_dim, act=paddle.activation.Softmax())
    return out
