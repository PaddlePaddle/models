import paddle.v2 as paddle

__all__ = ['alexnet']


def alexnet(input, class_dim):
    conv1 = paddle.layer.img_conv(
        input=input,
        filter_size=11,
        num_channels=3,
        num_filters=96,
        stride=4,
        padding=1)
    cmrnorm1 = paddle.layer.img_cmrnorm(
        input=conv1, size=5, scale=0.0001, power=0.75)
    pool1 = paddle.layer.img_pool(input=cmrnorm1, pool_size=3, stride=2)

    conv2 = paddle.layer.img_conv(
        input=pool1,
        filter_size=5,
        num_filters=256,
        stride=1,
        padding=2,
        groups=1)
    cmrnorm2 = paddle.layer.img_cmrnorm(
        input=conv2, size=5, scale=0.0001, power=0.75)
    pool2 = paddle.layer.img_pool(input=cmrnorm2, pool_size=3, stride=2)

    pool3 = paddle.networks.img_conv_group(
        input=pool2,
        pool_size=3,
        pool_stride=2,
        conv_num_filter=[384, 384, 256],
        conv_filter_size=3,
        pool_type=paddle.pooling.Max())

    fc1 = paddle.layer.fc(
        input=pool3,
        size=4096,
        act=paddle.activation.Relu(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5))
    fc2 = paddle.layer.fc(
        input=fc1,
        size=4096,
        act=paddle.activation.Relu(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5))

    out = paddle.layer.fc(
        input=fc2, size=class_dim, act=paddle.activation.Softmax())
    return out
