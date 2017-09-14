import paddle.v2 as paddle

__all__ = ['vgg13', 'vgg16', 'vgg19']


def vgg(input, nums, class_dim):
    def conv_block(input, num_filter, groups, num_channels=None):
        return paddle.networks.img_conv_group(
            input=input,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            pool_type=paddle.pooling.Max())

    assert len(nums) == 5
    # the channel of input feature is 3
    conv1 = conv_block(input, 64, nums[0], 3)
    conv2 = conv_block(conv1, 128, nums[1])
    conv3 = conv_block(conv2, 256, nums[2])
    conv4 = conv_block(conv3, 512, nums[3])
    conv5 = conv_block(conv4, 512, nums[4])

    fc_dim = 4096
    fc1 = paddle.layer.fc(
        input=conv5,
        size=fc_dim,
        act=paddle.activation.Relu(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5))
    fc2 = paddle.layer.fc(
        input=fc1,
        size=fc_dim,
        act=paddle.activation.Relu(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5))
    out = paddle.layer.fc(
        input=fc2, size=class_dim, act=paddle.activation.Softmax())
    return out


def vgg13(input, class_dim):
    nums = [2, 2, 2, 2, 2]
    return vgg(input, nums, class_dim)


def vgg16(input, class_dim):
    nums = [2, 2, 3, 3, 3]
    return vgg(input, nums, class_dim)


def vgg19(input, class_dim):
    nums = [2, 2, 4, 4, 4]
    return vgg(input, nums, class_dim)
