import paddle.v2 as paddle

__all__ = ['googlenet']


def inception(name, input, channels, filter1, filter3R, filter3, filter5R,
              filter5, proj):
    cov1 = paddle.layer.conv_projection(
        input=input,
        filter_size=1,
        num_channels=channels,
        num_filters=filter1,
        stride=1,
        padding=0)

    cov3r = paddle.layer.img_conv(
        name=name + '_3r',
        input=input,
        filter_size=1,
        num_channels=channels,
        num_filters=filter3R,
        stride=1,
        padding=0)
    cov3 = paddle.layer.conv_projection(
        input=cov3r, filter_size=3, num_filters=filter3, stride=1, padding=1)

    cov5r = paddle.layer.img_conv(
        name=name + '_5r',
        input=input,
        filter_size=1,
        num_channels=channels,
        num_filters=filter5R,
        stride=1,
        padding=0)
    cov5 = paddle.layer.conv_projection(
        input=cov5r, filter_size=5, num_filters=filter5, stride=1, padding=2)

    pool1 = paddle.layer.img_pool(
        name=name + '_max',
        input=input,
        pool_size=3,
        num_channels=channels,
        stride=1,
        padding=1)
    covprj = paddle.layer.conv_projection(
        input=pool1, filter_size=1, num_filters=proj, stride=1, padding=0)

    cat = paddle.layer.concat(
        name=name,
        input=[cov1, cov3, cov5, covprj],
        bias_attr=True,
        act=paddle.activation.Relu())
    return cat


def googlenet(input):
    # stage 1
    conv1 = paddle.layer.img_conv(
        name="conv1",
        input=input,
        filter_size=7,
        num_channels=3,
        num_filters=64,
        stride=2,
        padding=3)
    pool1 = paddle.layer.img_pool(
        name="pool1", input=conv1, pool_size=3, num_channels=64, stride=2)

    # stage 2
    conv2_1 = paddle.layer.img_conv(
        name="conv2_1",
        input=pool1,
        filter_size=1,
        num_filters=64,
        stride=1,
        padding=0)
    conv2_2 = paddle.layer.img_conv(
        name="conv2_2",
        input=conv2_1,
        filter_size=3,
        num_filters=192,
        stride=1,
        padding=1)
    pool2 = paddle.layer.img_pool(
        name="pool2", input=conv2_2, pool_size=3, num_channels=192, stride=2)

    # stage 3
    ince3a = inception("ince3a", pool2, 192, 64, 96, 128, 16, 32, 32)
    ince3b = inception("ince3b", ince3a, 256, 128, 128, 192, 32, 96, 64)
    pool3 = paddle.layer.img_pool(
        name="pool3", input=ince3b, num_channels=480, pool_size=3, stride=2)

    # stage 4
    ince4a = inception("ince4a", pool3, 480, 192, 96, 208, 16, 48, 64)
    ince4b = inception("ince4b", ince4a, 512, 160, 112, 224, 24, 64, 64)
    ince4c = inception("ince4c", ince4b, 512, 128, 128, 256, 24, 64, 64)
    ince4d = inception("ince4d", ince4c, 512, 112, 144, 288, 32, 64, 64)
    ince4e = inception("ince4e", ince4d, 528, 256, 160, 320, 32, 128, 128)
    pool4 = paddle.layer.img_pool(
        name="pool4", input=ince4e, num_channels=832, pool_size=3, stride=2)

    # stage 5
    ince5a = inception("ince5a", pool4, 832, 256, 160, 320, 32, 128, 128)
    ince5b = inception("ince5b", ince5a, 832, 384, 192, 384, 48, 128, 128)
    pool5 = paddle.layer.img_pool(
        name="pool5",
        input=ince5b,
        num_channels=1024,
        pool_size=7,
        stride=7,
        pool_type=paddle.pooling.Avg())
    dropout = paddle.layer.addto(
        input=pool5,
        layer_attr=paddle.attr.Extra(drop_rate=0.4),
        act=paddle.activation.Linear())

    # fc for output 1
    pool_o1 = paddle.layer.img_pool(
        name="pool_o1",
        input=ince4a,
        num_channels=512,
        pool_size=5,
        stride=3,
        pool_type=paddle.pooling.Avg())
    conv_o1 = paddle.layer.img_conv(
        name="conv_o1",
        input=pool_o1,
        filter_size=1,
        num_filters=128,
        stride=1,
        padding=0)
    fc_o1 = paddle.layer.fc(
        name="fc_o1",
        input=conv_o1,
        size=1024,
        layer_attr=paddle.attr.Extra(drop_rate=0.7),
        act=paddle.activation.Relu())

    # fc for output 2
    pool_o2 = paddle.layer.img_pool(
        name="pool_o2",
        input=ince4d,
        num_channels=528,
        pool_size=5,
        stride=3,
        pool_type=paddle.pooling.Avg())
    conv_o2 = paddle.layer.img_conv(
        name="conv_o2",
        input=pool_o2,
        filter_size=1,
        num_filters=128,
        stride=1,
        padding=0)
    fc_o2 = paddle.layer.fc(
        name="fc_o2",
        input=conv_o2,
        size=1024,
        layer_attr=paddle.attr.Extra(drop_rate=0.7),
        act=paddle.activation.Relu())

    return dropout, fc_o1, fc_o2
