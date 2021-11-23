import paddle.v2 as paddle

__all__ = ['drn16']

def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  active_type=paddle.activation.Relu(),
                  ch_in=None):
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False)
    return paddle.layer.batch_norm(input=tmp, act=active_type)


def shortcut(input, ch_out, stride):
    if input.num_filters != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0,
                             paddle.activation.Linear())
    else:
        return input


def basicblock(input, ch_out, stride):
    short = shortcut(input, ch_out, stride)
    conv1 = conv_bn_layer(input, ch_out, 3, stride, 1)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, paddle.activation.Linear())
    return paddle.layer.addto(
        input=[short, conv2], act=paddle.activation.Relu())


def bottleneck(input, ch_out, stride):
    short = shortcut(input, ch_out * 4, stride)
    conv1 = conv_bn_layer(input, ch_out, 1, stride, 0)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1)
    conv3 = conv_bn_layer(conv2, ch_out * 4, 1, 1, 0,
                          paddle.activation.Linear())
    return paddle.layer.addto(
        input=[short, conv3], act=paddle.activation.Relu())


def layer_warp(block_func, input, ch_out, count, stride):
    conv = block_func(input, ch_out, stride)
    for i in range(1, count):
        conv = block_func(conv, ch_out, 1)
    return conv


def drn16(input, class_dim, depth=32):
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    conv1 = conv_bn_layer(
        input, ch_in=3, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = paddle.layer.img_pool(
        input=res3, pool_size=8, stride=1, pool_type=paddle.pooling.Avg())
    out = paddle.layer.fc(input=pool,
                          size=class_dim,
                          act=paddle.activation.Softmax())
    return out