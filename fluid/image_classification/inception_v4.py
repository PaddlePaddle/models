import os
import paddle.fluid as fluid


def inception_v4(image, label):

    tmp = stem(input=image)
    for i in range(0, 4):
        tmp = inception_A(input=tmp, depth=i)
    tmp = reduction_A(input=tmp)

    for i in range(0, 7):
        tmp = inception_B(input=tmp, depth=i)
    reduction_B(input=tmp)

    for i in range(0, 3):
        tmp = inception_C(input=tmp, depth=i)

    pool = fluid.layers.pool2d(
        pool_type='ave', input=tmp, pool_size=7, pool_stride=1)
    dropout = fluid.layers.dropout(input=pool, drop_prob=0.2)
    out = fluid.layers.softmax(input=dropout)
    return out


def conv_bn_layer(input,
                  num_filters,
                  filter_size,
                  padding=1,
                  stride=1,
                  groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=groups,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def stem(input):
    conv1 = conv_bn_layer(input=input, num_filters=32, filter_size=3, stride=2)
    conv2 = conv_bn_layer(input=conv1, num_filters=32, filter_size=3)
    conv3 = conv_bn_layer(input=conv2, num_filters=64, filter_size=3)

    def block0(input):
        pool0 = fluid.layers.pool2d(
            input=input, pool_size=3, pool_stride=2, pool_type='max')
        conv0 = conv_bn_layer(
            input=input, num_filters=96, filter_size=3, stride=2)
        return fluid.layers.concat(input=[pool0, conv0])

    def block1(input):
        l_conv0 = conv_bn_layer(
            input=input, num_filters=64, filter_size=1, stride=1, padding=0)
        l_conv1 = conv_bn_layer(
            input=l_conv0, num_filters=96, filter_size=3, stride=1, padding=1)
        r_conv0 = conv_bn_layer(
            input=input, num_filters=64, filter_size=1, stride=1, padding=0)
        r_conv1 = conv_bn_layer(
            input=r_conv0,
            num_filters=64,
            filter_size=(7, 1),
            stride=1,
            padding=(3, 0))
        r_conv2 = conv_bn_layer(
            input=r_conv1,
            num_filters=64,
            filter_size=(1, 7),
            stride=1,
            padding=(0, 3))
        r_conv3 = conv_bn_layer(
            input=r_conv2, num_filters=96, filter_size=3, stride=1, padding=1)
        return fluid.layers.concat(input=[l_conv1, r_conv3])

    def block2(input):
        conv0 = conv_bn_layer(
            input=input, num_filters=192, filter_size=3, stride=2, padding=1)
        pool0 = fluid.layers.pool2d(
            input=input, pool_size=3, pool_stride=2, pool_type='max')
        return fluid.layers.concat(input=[conv0, pool0])

    conv3 = block0(conv2)
    conv4 = block1(conv3)
    conv5 = block2(conv4)
    return conv5


def inception_A(input, depth):
    b0_pool0 = paddle.layer.pool2d(
        name='inceptA{0}_branch0_pool0'.format(depth),
        input=input,
        pool_size=3,
        stride=1,
        padding=1,
        pool_type='avg')
    b0_conv0 = conv_bn_layer(
        name='inceptA{0}_branch0_conv0'.format(depth),
        input=b0_pool0,
        num_filters=96,
        filter_size=1,
        stride=1,
        padding=0)
    b1_conv0 = conv_bn_layer(
        name='inceptA{0}_branch1_conv0'.format(depth),
        input=input,
        num_filters=96,
        filter_size=1,
        stride=1,
        padding=0)
    b2_conv0 = conv_bn_layer(
        name='inceptA{0}_branch2_conv0'.format(depth),
        input=input,
        num_filters=64,
        filter_size=1,
        stride=1,
        padding=0)
    b2_conv1 = conv_bn_layer(
        name='inceptA{0}_branch2_conv1'.format(depth),
        input=b2_conv0,
        num_channels=64,
        num_filters=96,
        filter_size=3,
        stride=1,
        padding=1)
    b3_conv0 = conv_bn_layer(
        name='inceptA{0}_branch3_conv0'.format(depth),
        input=input,
        num_channels=384,
        num_filters=64,
        filter_size=1,
        stride=1,
        padding=0)
    b3_conv1 = conv_bn_layer(
        name='inceptA{0}_branch3_conv1'.format(depth),
        input=b3_conv0,
        num_filters=96,
        filter_size=3,
        stride=1,
        padding=1)
    b3_conv2 = conv_bn_layer(
        name='inceptA{0}_branch3_conv2'.format(depth),
        input=b3_conv1,
        num_filters=96,
        filter_size=3,
        stride=1,
        padding=1)
    return paddle.layer.concat(input=[b0_conv0, b1_conv0, b2_conv1, b3_conv2])


def reduction_A(input):
    b0_pool0 = fluid.layers.pool2d(
        name='ReductA_branch0_pool0',
        input=input,
        pool_size=3,
        pool_stride=2,
        pool_type='max')
    b1_conv0 = conv_bn_layer(
        name='ReductA_branch1_conv0',
        input=input,
        num_filters=384,
        filter_size=3,
        stride=2,
        padding=1)
    b2_conv0 = conv_bn_layer(
        name='ReductA_branch2_conv0',
        input=input,
        num_filters=192,
        filter_size=1,
        stride=1,
        padding=0)
    b2_conv1 = conv_bn_layer(
        name='ReductA_branch2_conv1',
        input=b2_conv0,
        num_filters=224,
        filter_size=3,
        stride=1,
        padding=1)
    b2_conv2 = conv_bn_layer(
        name='ReductA_branch2_conv2',
        input=b2_conv1,
        num_filters=256,
        filter_size=3,
        stride=2,
        padding=1)
    return fluid.layers.concat(input=[b0_pool0, b1_conv0, b2_conv2])


def inception_B(input, depth):
    b0_pool0 = fluid.layers.pool2d(
        name='inceptB{0}_branch0_pool0'.format(depth),
        input=input,
        pool_size=3,
        pool_stride=1,
        pool_padding=1,
        pool_type='avg')
    b0_conv0 = conv_bn_layer(
        name='inceptB{0}_branch0_conv0'.format(depth),
        input=b0_pool0,
        num_filters=128,
        filter_size=1,
        stride=1,
        padding=0)
    b1_conv0 = conv_bn_layer(
        name='inceptB{0}_branch1_conv0'.format(depth),
        input=input,
        num_filters=384,
        filter_size=1,
        stride=1,
        padding=0)
    b2_conv0 = conv_bn_layer(
        name='inceptB{0}_branch2_conv0'.format(depth),
        input=input,
        num_filters=192,
        filter_size=1,
        stride=1,
        padding=0)
    b2_conv1 = conv_bn_layer(
        name='inceptB{0}_branch2_conv1'.format(depth),
        input=b2_conv0,
        num_filters=224,
        filter_size=(1, 7),
        stride=1,
        padding=(0, 3))
    b2_conv2 = conv_bn_layer(
        name='inceptB{0}_branch2_conv2'.format(depth),
        input=b2_conv1,
        num_filters=256,
        filter_size=(7, 1),
        stride=1,
        padding=(3, 0))
    b3_conv0 = conv_bn_layer(
        name='inceptB{0}_branch3_conv0'.format(depth),
        input=input,
        num_filters=192,
        filter_size=1,
        stride=1,
        padding=0)
    b3_conv1 = conv_bn_layer(
        name='inceptB{0}_branch3_conv1'.format(depth),
        input=b3_conv0,
        num_filters=192,
        filter_size=(1, 7),
        stride=1,
        padding=(0, 3))
    b3_conv2 = conv_bn_layer(
        name='inceptB{0}_branch3_conv2'.format(depth),
        input=b3_conv1,
        num_filters=224,
        filter_size=(7, 1),
        stride=1,
        padding=(3, 0))
    b3_conv3 = conv_bn_layer(
        name='inceptB{0}_branch3_conv3'.format(depth),
        input=b3_conv2,
        num_filters=224,
        filter_size=(1, 7),
        stride=1,
        padding=(0, 3))
    b3_conv4 = conv_bn_layer(
        name='inceptB{0}_branch3_conv4'.format(depth),
        input=b3_conv3,
        num_filters=256,
        filter_size=(7, 1),
        stride=1,
        padding=(3, 0))
    return fluid.layers.concat(input=[b0_conv0, b1_conv0, b2_conv2, b3_conv4])


def reduction_B(input):
    b0_pool0 = fluid.layers.pool2d(
        name='ReductB_branch0_pool0',
        input=input,
        pool_size=3,
        pool_stride=2,
        pool_type='max')
    b1_conv0 = conv_bn_layer(
        name='ReductB_branch1_conv0',
        input=input,
        num_filters=192,
        filter_size=1,
        stride=1,
        padding=0)
    b1_conv1 = conv_bn_layer(
        name='ReductB_branch1_conv1',
        input=b1_conv0,
        num_filters=192,
        filter_size=3,
        stride=2,
        padding=1)
    b2_conv0 = conv_bn_layer(
        name='ReductB_branch2_conv0',
        input=input,
        num_filters=256,
        filter_size=1,
        stride=1,
        padding=0)
    b2_conv1 = conv_bn_layer(
        name='ReductB_branch2_conv1',
        input=b2_conv0,
        num_filters=256,
        filter_size=(1, 7),
        stride=1,
        padding=(0, 3))
    b2_conv2 = conv_bn_layer(
        name='ReductB_branch2_conv2',
        input=b2_conv1,
        num_filters=320,
        filter_size=(7, 1),
        stride=1,
        padding=(3, 0))
    b2_conv3 = conv_bn_layer(
        name='ReductB_branch2_conv3',
        input=b2_conv2,
        num_filters=320,
        filter_size=3,
        stride=2,
        padding=1)
    return fluid.layers.concat(input=[b0_pool0, b1_conv1, b2_conv3])


def inception_C(input, depth):
    b0_pool0 = fluid.layers.pool2d(
        name='inceptC{0}_branch0_pool0'.format(depth),
        input=input,
        pool_size=3,
        pool_stride=1,
        pool_padding=1,
        pool_type='avg')
    b0_conv0 = conv_bn_layer(
        name='inceptC{0}_branch0_conv0'.format(depth),
        input=b0_pool0,
        num_filters=256,
        filter_size=1,
        stride=1,
        padding=0)
    b1_conv0 = conv_bn_layer(
        name='inceptC{0}_branch1_conv0'.format(depth),
        input=input,
        num_filters=256,
        filter_size=1,
        stride=1,
        padding=0)
    b2_conv0 = conv_bn_layer(
        name='inceptC{0}_branch2_conv0'.format(depth),
        input=input,
        num_filters=384,
        filter_size=1,
        stride=1,
        padding=0)
    b2_conv1 = conv_bn_layer(
        name='inceptC{0}_branch2_conv1'.format(depth),
        input=b2_conv0,
        num_filters=256,
        filter_size=(1, 3),
        stride=1,
        padding=(0, 1))
    b2_conv2 = conv_bn_layer(
        name='inceptC{0}_branch2_conv2'.format(depth),
        input=b2_conv0,
        num_filters=256,
        filter_size=(3, 1),
        stride=1,
        padding=(1, 0))
    b3_conv0 = conv_bn_layer(
        name='inceptC{0}_branch3_conv0'.format(depth),
        input=input,
        num_filters=384,
        filter_size=1,
        stride=1,
        padding=0)
    b3_conv1 = conv_bn_layer(
        name='inceptC{0}_branch3_conv1'.format(depth),
        input=b3_conv0,
        num_filters=448,
        filter_size=(1, 3),
        stride=1,
        padding=(0, 1))
    b3_conv2 = conv_bn_layer(
        name='inceptC{0}_branch3_conv2'.format(depth),
        input=b3_conv1,
        num_filters=512,
        filter_size=(3, 1),
        stride=1,
        padding=(1, 0))
    b3_conv3 = conv_bn_layer(
        name='inceptC{0}_branch3_conv3'.format(depth),
        input=b3_conv2,
        num_filters=256,
        filter_size=(3, 1),
        stride=1,
        padding=(1, 0))
    b3_conv4 = conv_bn_layer(
        name='inceptC{0}_branch3_conv4'.format(depth),
        input=b3_conv2,
        num_filters=256,
        filter_size=(1, 3),
        stride=1,
        padding=(0, 1))
    return fluid.layers.concat(
        input=[b0_conv0, b1_conv0, b2_conv1, b2_conv2, b3_conv3, b3_conv4])
